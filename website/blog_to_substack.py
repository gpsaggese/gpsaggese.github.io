#!/usr/bin/env python
"""
Convert MkDocs blog posts into HTML ready to paste into the Substack editor.

The Substack editor keeps formatting only from HTML on the clipboard, so the
script converts the Markdown with pandoc, applies the fixes Substack needs (see
`website/blog_to_substack.lua`), adds an "Originally published by" line with the
authors' GitHub handles, and either copies the result to the clipboard or writes
a page with "Copy" buttons.

See `website/blog_to_substack.how_to_guide.md` for the full process.

# Usage

- Convert one post and copy it to the clipboard (macOS or Linux):
> website/blog_to_substack.py --file website/docs/blog/posts/how_to.Use_OpenRouter.md --copy

- Convert one post into a page with Copy buttons:
> website/blog_to_substack.py --file website/docs/blog/posts/how_to.Use_OpenRouter.md

- Convert all the published posts into one page with Copy buttons:
> website/blog_to_substack.py --all

- Render tables as images instead of code blocks (needs Playwright):
> website/blog_to_substack.py --file ... --tables png
"""

import argparse
import base64
import glob
import html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import yaml
from pymdownx.slugs import slugify

_LOG = logging.getLogger(__name__)

_POSTS_DIR = "website/docs/blog/posts"
_AUTHORS_FILE = "website/docs/.authors.yml"
_SITE_URL = "https://gpsaggese.github.io/blog/"
_LUA_FILTER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "blog_to_substack.lua"
)

# #############################################################################
# Front matter and authors
# #############################################################################


def _split_front_matter(text: str) -> Tuple[Dict, str]:
    """
    Return the YAML front matter as a dict and the body without it.
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, text[m.end() :]


def _load_authors() -> Dict:
    if not os.path.exists(_AUTHORS_FILE):
        _LOG.warning(
            "Can't find '%s': authors will be shown by handle only",
            _AUTHORS_FILE,
        )
        return {}
    with open(_AUTHORS_FILE, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("authors", {})


def _post_url(title: str) -> str:
    """
    Return the URL of the post on the website.

    The MkDocs blog plugin builds it from the title with this slugify.
    """
    return _SITE_URL + slugify(case="lower")(title, "-") + "/"


def _attribution_html(authors: List[str], url: str, all_authors: Dict) -> str:
    """
    Build the first line of the Substack post, e.g.,

    Originally published on gpsaggese.github.io by Giacinto Paolo Saggese (@gpsaggese)
    """
    parts = []
    for handle in authors:
        info = all_authors.get(handle, {})
        name = info.get("name", handle)
        if " " in handle:
            # Not a GitHub handle, e.g., a free-form name.
            parts.append(html.escape(name))
        else:
            gh = f"https://github.com/{handle}"
            parts.append(
                f'{html.escape(name)} (<a href="{gh}">@{html.escape(handle)}</a>)'
            )
    who = ", ".join(parts) if parts else "the GP Research team"
    return (
        f'<p><em>Originally published on <a href="{url}">gpsaggese.github.io</a> '
        f"by {who}.</em></p>\n"
    )


# #############################################################################
# Conversion
# #############################################################################


def _run_pandoc(body: str, tables: str) -> str:
    cmd = [
        "pandoc",
        "-f",
        "gfm-tex_math_dollars",
        "-t",
        "html5",
        "--no-highlight",
        "--wrap=none",
        "--metadata",
        f"substack_tables={tables}",
        "--lua-filter",
        _LUA_FILTER,
    ]
    return subprocess.run(
        cmd, input=body, capture_output=True, text=True, check=True
    ).stdout


def _url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _fix_images(frag: str, check_live: bool) -> Tuple[str, List[str]]:
    """
    Point relative image paths to the live website, where Substack can fetch them.

    :return: the HTML and the list of images that are not on the live website
    """
    missing = []

    def _repl(m: re.Match) -> str:
        path = re.sub(r"^\./", "", m.group(1))
        url = _SITE_URL + path
        if check_live and not _url_exists(url):
            missing.append(url)
        return f'src="{url}"'

    frag = re.sub(r'src="(?!https?://|data:)([^"]+)"', _repl, frag)
    return frag, missing


def _tables_to_png(frag: str, png_dir: str) -> str:
    """
    Replace each HTML table with a PNG screenshot, embedded as a data URI.

    The PNG files are also saved in `png_dir`, to drag into Substack by hand if
    the editor drops embedded images.
    """
    tables = re.findall(r"<table.*?</table>", frag, re.DOTALL)
    if not tables:
        return frag
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _LOG.error(
            "--tables png needs Playwright: pip install playwright && playwright install chromium"
        )
        sys.exit(1)
    os.makedirs(png_dir, exist_ok=True)
    css = (
        "body{margin:0;background:#fff;font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif;color:#222}"
        "table{border-collapse:collapse;margin:8px}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}"
        "th{background:#f3f3f3}code{font-family:Menlo,monospace;font-size:13px;background:#f3f3f3;padding:1px 3px}"
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        for i, tbl in enumerate(tables, 1):
            page.set_content(
                f"<html><head><style>{css}</style></head><body>{tbl}</body></html>"
            )
            png = page.locator("table").screenshot()
            path = os.path.join(png_dir, f"table_{i}.png")
            with open(path, "wb") as f:
                f.write(png)
            alt = html.escape(
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", tbl)).strip()
            )[:500]
            uri = "data:image/png;base64," + base64.b64encode(png).decode()
            frag = frag.replace(tbl, f'<p><img src="{uri}" alt="{alt}"></p>', 1)
            _LOG.info("Table %d -> %s", i, path)
        browser.close()
    return frag


def convert_post(
    file: str, tables: str, check_live: bool, out_dir: str, all_authors: Dict
) -> Dict:
    """
    Convert one blog post.

    :return: dict with title, subtitle, date, url, html and warnings
    """
    with open(file, encoding="utf-8") as f:
        fm, body = _split_front_matter(f.read())
    title = str(fm.get("title", os.path.basename(file)))
    url = _post_url(title)
    authors = fm.get("authors") or []
    frag = _run_pandoc(body, tables)
    frag = re.sub(r"</?span>", "", frag)
    warnings = []
    frag, missing = _fix_images(frag, check_live)
    if missing:
        warnings.append(
            f"{len(missing)} image(s) not on the live website, upload by hand: "
            + ", ".join(missing)
        )
    if tables == "png":
        png_dir = os.path.join(
            out_dir, os.path.basename(file).replace(".md", "") + ".tables"
        )
        frag = _tables_to_png(frag, png_dir)
    if check_live and not _url_exists(url):
        warnings.append(f"post not on the live website yet: {url}")
    frag = _attribution_html(authors, url, all_authors) + frag
    return {
        "file": os.path.basename(file),
        "title": title,
        "subtitle": str(fm.get("description") or ""),
        "date": str(fm.get("date", "")),
        "url": url,
        "html": frag,
        "code": frag.count("<pre"),
        "images": frag.count("<img"),
        "warnings": warnings,
    }


# #############################################################################
# Output
# #############################################################################


def _copy_to_clipboard(frag: str) -> None:
    """
    Put `frag` on the clipboard as HTML, so Substack keeps the formatting.
    """
    if sys.platform == "darwin":
        script = (
            "ObjC.import('AppKit');"
            "var s=$.NSString.alloc.initWithUTF8String(%s);"
            "var pb=$.NSPasteboard.generalPasteboard;pb.clearContents;"
            "pb.setStringForType(s,$.NSPasteboardTypeHTML);"
        ) % json.dumps(frag)
        subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            check=True,
            capture_output=True,
        )
    elif shutil.which("xclip"):
        subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "text/html"],
            input=frag,
            text=True,
            check=True,
        )
    elif shutil.which("wl-copy"):
        subprocess.run(
            ["wl-copy", "--type", "text/html"], input=frag, text=True, check=True
        )
    else:
        _LOG.error(
            "No clipboard tool found: run without --copy and use the Copy buttons"
        )
        sys.exit(1)


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Substack copy</title>
<style>body{font:15px/1.5 -apple-system,Helvetica,Arial,sans-serif;max-width:860px;margin:24px auto;padding:0 16px}
.post{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:12px 0}.meta{color:#666;font-size:13px}
.warn{color:#b45309;font-size:13px}button{margin:8px 6px 0 0;padding:5px 11px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer}
button.main{background:#ff6719;color:#fff;border-color:#ff6719}.preview{display:none;border-top:1px solid #eee;margin-top:10px;padding-top:10px}
.preview pre{background:#f3f3f3;padding:8px;overflow:auto}.preview img{max-width:100%}</style></head><body>
<h1>Substack copy</h1><p>In Substack: Create &rarr; Article. Copy title &rarr; paste in the title. Copy subtitle &rarr; paste in the subtitle.
Click into the body, Copy body &rarr; Cmd+V.</p><div id="list"></div>
<script>
const POSTS = __DATA__;
function copyRich(h,p,b){const f=e=>{e.clipboardData.setData('text/html',h);e.clipboardData.setData('text/plain',p);e.preventDefault()};
document.addEventListener('copy',f);document.execCommand('copy');document.removeEventListener('copy',f);const t=b.textContent;b.textContent='Copied';setTimeout(()=>b.textContent=t,1200)}
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
POSTS.forEach((p,i)=>{const d=document.createElement('div');d.className='post';
d.innerHTML=`<b>${i+1}. ${esc(p.title)}</b><div class="meta">${p.date} &middot; ${p.file} &middot; ${p.code} code blocks &middot; ${p.images} images &middot; <a href="${p.url}">live post</a></div>`+
p.warnings.map(w=>`<div class="warn">&#9888; ${esc(w)}</div>`).join('')+
`<button data-a="t">Copy title</button>${p.subtitle?'<button data-a="s">Copy subtitle</button>':''}<button class="main" data-a="b">Copy body</button><button data-a="p">Preview</button><div class="preview">${p.html}</div>`;
d.querySelectorAll('button').forEach(b=>b.onclick=()=>{const a=b.dataset.a;if(a=='t')copyRich(esc(p.title),p.title,b);else if(a=='s')copyRich(esc(p.subtitle),p.subtitle,b);
else if(a=='b'){const x=document.createElement('div');x.innerHTML=p.html;copyRich(p.html,x.innerText,b)}else d.querySelector('.preview').style.display=d.querySelector('.preview').style.display=='block'?'none':'block'});
document.getElementById('list').appendChild(d)});
</script></body></html>
"""


def _write_page(posts: List[Dict], path: str) -> None:
    data = json.dumps(posts).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_PAGE.replace("__DATA__", data))
    _LOG.info("Wrote %s: open it in Chrome", path)


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Blog post to convert")
    group.add_argument(
        "--all",
        action="store_true",
        help="Convert all published posts (no `draft.` prefix)",
    )
    parser.add_argument(
        "--tables",
        choices=["code", "png"],
        default="code",
        help="Render tables as code blocks or images",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the body to the clipboard (only with --file)",
    )
    parser.add_argument(
        "--out_dir",
        default="tmp.build/substack",
        help="Where to write the page and the table images",
    )
    parser.add_argument(
        "--no_check_live",
        action="store_true",
        help="Don't check the live website for posts and images",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if not os.path.isdir(_POSTS_DIR):
        parser.error(f"Run from the repo root: can't find '{_POSTS_DIR}'")
    if args.copy and not args.file:
        parser.error("--copy works only with --file")
    if not shutil.which("pandoc"):
        parser.error("pandoc not found: install it (e.g., brew install pandoc)")
    os.makedirs(args.out_dir, exist_ok=True)
    all_authors = _load_authors()
    if args.file:
        files = [args.file]
    else:
        files = sorted(
            f
            for f in glob.glob(os.path.join(_POSTS_DIR, "*.md"))
            if not os.path.basename(f).startswith("draft.")
            and os.path.basename(f) != "Welcome_to_Our_Blog.md"
        )
    posts = [
        convert_post(
            f, args.tables, not args.no_check_live, args.out_dir, all_authors
        )
        for f in files
    ]
    posts.sort(key=lambda p: p["date"])
    for p in posts:
        print(f"{p['file']}: {p['code']} code blocks, {p['images']} images")
        for w in p["warnings"]:
            print(f"  WARNING: {w}")
    if args.copy:
        p = posts[0]
        _copy_to_clipboard(p["html"])
        print("\nBody copied to the clipboard. In Substack, also set:")
        print(f"  Title:    {p['title']}")
        print(f"  Subtitle: {p['subtitle']}")
        print(f"  Date:     {p['date']} (post settings, to match the original)")
    else:
        name = "all_posts" if args.all else posts[0]["file"].replace(".md", "")
        _write_page(posts, os.path.join(args.out_dir, f"{name}.html"))
        print(
            f"\nOpen {os.path.join(args.out_dir, name + '.html')} in Chrome and use the Copy buttons"
        )


if __name__ == "__main__":
    _main(_parse())
