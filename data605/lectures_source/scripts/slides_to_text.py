import argparse
import logging
import os
import re
import sys
import urllib.request
import urllib.parse
from typing import List, Optional, Tuple
import io
import zipfile
import json
import time

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp
except Exception as exc:  # ImportError or other issues
    # Defer raising until main entrypoint to allow --help to work without deps
    GOOGLE_IMPORT_ERROR: Optional[Exception] = exc
else:
    GOOGLE_IMPORT_ERROR = None


SCOPES = [
    "https://www.googleapis.com/auth/presentations.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
CREDENTIALS_FILE = "credentials.json"  # Expected to be in the working directory
TOKEN_FILE = "token.json"


def configure_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity <= 0:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize_filename(name: str, max_length: int = 120) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3].rstrip() + "..."
    return sanitized or "presentation"


def extract_presentation_id(url_or_id: str) -> Optional[str]:
    # Accept raw ID or any of common Google Slides URL variants
    url = url_or_id.strip()
    if not url:
        return None

    # If it's already an ID-like token
    if re.fullmatch(r"[a-zA-Z0-9_-]+", url):
        return url

    # Common patterns for Google Slides URLs
    patterns = [
        r"https?://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)

    return None


def authenticate_and_get_creds() -> "Credentials":
    """Handles user authentication and returns credentials."""
    if GOOGLE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Google API libraries are not available: " f"{GOOGLE_IMPORT_ERROR}"
        )

    creds: Optional[Credentials] = None

    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as exc:
            logging.warning("Failed to load existing token: %s", exc)
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logging.warning("Token refresh failed, falling back to new flow: %s", exc)
                # If refresh fails, delete the token and re-authenticate
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                creds = None

        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE}. Download OAuth client credentials (Desktop app) "
                    "from Google Cloud Console and place the JSON file alongside this script."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        try:
            with open(TOKEN_FILE, "w", encoding="utf-8") as token:
                token.write(creds.to_json())
        except Exception as exc:
            logging.warning("Failed to write token file: %s", exc)

    return creds


def sanitize_lecture_number_for_filename(lecture_number: float) -> str:
    """Convert lecture number to a filename-safe string."""
    return str(lecture_number).replace('.', '_')


def download_images_in_order(creds: "Credentials", slide_data: List[dict], images_dir: str, lecture_number: float) -> dict:
    """
    Downloads images directly from the API in the order they appear in the slides.
    Returns a dictionary mapping image object IDs to their local file paths.
    """
    image_map = {}
    http = AuthorizedHttp(creds)

    for slide_index, slide in enumerate(slide_data):
        slide_image_counter = 1
        for element in slide.get("elements", []):
            if element.get("type") == "image":
                object_id = element.get("object_id")
                image_props = element.get("image_properties", {})
                content_url = image_props.get("contentUrl")

                if content_url:
                    try:
                        response, content = http.request(content_url)
                        if response.status == 200:
                            # Guess extension from content-type or default to png
                            content_type = response.get("content-type", "image/png")
                            extension = f".{content_type.split('/')[-1]}"
                            
                            lecture_safe = sanitize_lecture_number_for_filename(lecture_number)
                            image_filename = f"lec_{lecture_safe}_slide_{slide_index + 1}_image_{slide_image_counter}{extension}"
                            image_path = os.path.join(images_dir, image_filename)
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(content)
                            
                            relative_path = os.path.join(
                                "images", f"lecture_{lecture_safe}", image_filename
                            )
                            image_map[object_id] = relative_path
                            logging.info(f"Downloaded image: {relative_path}")
                            
                            slide_image_counter += 1
                            time.sleep(0.5)  # Avoid rate limiting
                        else:
                            logging.warning(f"Failed to download image from {content_url}, status: {response.status}")
                            image_map[object_id] = "image_download_failed"
                    except Exception as e:
                        logging.error(f"Error downloading image from {content_url}: {e}")
                        image_map[object_id] = "image_download_failed"
    return image_map


def extract_text_with_formatting(text_elements: List[dict]) -> List[dict]:
    """Extract text with formatting information, preserving paragraph structure."""
    if not text_elements:
        return []

    paragraphs = []
    # Start with a default paragraph, in case the text doesn't begin with a paragraphMarker.
    current_paragraph = {"type": "paragraph", "bullet": None, "content_parts": []}

    for element in text_elements:
        if "paragraphMarker" in element:
            # If the current paragraph has content, store it before starting a new one.
            if any(part.get("content", "").strip() for part in current_paragraph["content_parts"]):
                paragraphs.append(current_paragraph)

            bullet_info = element.get("paragraphMarker", {}).get("bullet")
            current_paragraph = {
                "type": "paragraph",
                "bullet": None,
                "content_parts": []
            }
            if bullet_info:
                current_paragraph["bullet"] = {
                    "nestingLevel": bullet_info.get("nestingLevel", 0),
                    "glyph": bullet_info.get("glyph", "*")
                }

        text_run = element.get("textRun")
        auto_text = element.get("autoText")

        if text_run:
            content = text_run.get("content", "")
            style = text_run.get("style", {})
            current_paragraph["content_parts"].append({
                "type": "text",
                "content": content,
                "bold": style.get("bold", False),
                "italic": style.get("italic", False),
                "underline": style.get("underline", False),
                "font_size": style.get("fontSize", {}).get("magnitude", None),
                "font_family": style.get("fontFamily", None),
                "foreground_color": style.get("foregroundColor", {}).get("opaqueColor", {}).get("rgbColor", {}),
                "link": style.get("link", {}).get("url", None)
            })
        elif auto_text:
            content = auto_text.get("content", "")
            current_paragraph["content_parts"].append({
                "type": "auto_text",
                "content": content,
                "auto_text_type": auto_text.get("type", "UNSPECIFIED")
            })

    # Add the last paragraph if it has content
    if any(part.get("content", "").strip() for part in current_paragraph["content_parts"]):
        paragraphs.append(current_paragraph)

    return paragraphs


def extract_plain_text_from_paragraphs(paragraphs: List[dict]) -> str:
    """Extracts plain text from a list of paragraph structures."""
    full_text_parts = []
    for para in paragraphs:
        for part in para.get("content_parts", []):
            full_text_parts.append(part.get("content", ""))
    
    full = "".join(full_text_parts)
    # Normalize whitespace and remove blank-only lines
    lines = [ln.strip("\u00A0 \t") for ln in full.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_table_structure(table: dict) -> dict:
    """Extract table with structure and formatting information."""
    n_rows = table.get("rows", 0)
    n_cols = table.get("columns", 0)
    
    table_data = {
        "type": "table",
        "rows": n_rows,
        "columns": n_cols,
        "cells": []
    }
    
    for r in range(n_rows):
        row_cells = table.get("tableRows", [])[r : r + 1]
        if not row_cells:
            continue
        row = row_cells[0]
        
        for c in range(n_cols):
            cells = row.get("tableCells", [])[c : c + 1]
            if not cells:
                table_data["cells"].append({
                    "row": r,
                    "col": c,
                    "content": "",
                    "formatted_content": []
                })
                continue
                
            cell = cells[0]
            paras = cell.get("text", {}).get("textElements")
            formatted_content = extract_text_with_formatting(paras or [])
            cell_text = extract_plain_text_from_paragraphs(formatted_content)
            
            # Check if this is a header cell (usually first row or has special styling)
            is_header = (r == 0) or cell.get("tableCellProperties", {}).get("tableCellBackgroundFill", {})
            
            table_data["cells"].append({
                "row": r,
                "col": c,
                "content": cell_text,
                "formatted_content": formatted_content,
                "is_header": is_header,
                "cell_properties": cell.get("tableCellProperties", {})
            })
    
    return table_data


def extract_text_from_table(table: dict) -> str:
    """Legacy function for plain text table extraction."""
    # Combine cell texts row-by-row, separating cells with a tab
    n_rows = table.get("rows", 0)
    n_cols = table.get("columns", 0)
    rows_text: List[str] = []
    for r in range(n_rows):
        cells_text: List[str] = []
        row_cells = table.get("tableRows", [])[r : r + 1]
        if not row_cells:
            continue
        row = row_cells[0]
        for c in range(n_cols):
            cells = row.get("tableCells", [])[c : c + 1]
            if not cells:
                cells_text.append("")
                continue
            cell = cells[0]
            paras = cell.get("text", {}).get("textElements")
            cell_text = extract_plain_text_from_paragraphs(extract_text_with_formatting(paras or []))
            cells_text.append(cell_text)
        rows_text.append("\t".join(cells_text).strip())
    return "\n".join([rt for rt in rows_text if rt])


def extract_slide_elements(slide: dict) -> List[dict]:
    """Extract all slide elements with their structure and formatting."""
    elements = []
    
    for element in slide.get("pageElements", []):
        element_data = {
            "object_id": element.get("objectId"),
            "transform": element.get("transform", {}),
            "size": element.get("size", {})
        }
        
        # Handle shapes (text boxes, titles, etc.)
        if "shape" in element:
            shape = element["shape"]
            placeholder = shape.get("placeholder")
            
            if placeholder:
                element_data.update({
                    "type": "placeholder",
                    "placeholder_type": placeholder.get("type", "UNSPECIFIED"),
                    "index": placeholder.get("index", 0)
                })
            
            # Extract text content with formatting
            text_elements = shape.get("text", {}).get("textElements", [])
            if text_elements:
                element_data.update({
                    "type": "text_shape",
                    "content": extract_plain_text_from_paragraphs(extract_text_with_formatting(text_elements)),
                    "formatted_content": extract_text_with_formatting(text_elements)
                })
            
            # Check for images
            if shape.get("shapeProperties", {}).get("shapeFill", {}).get("pictureFill"):
                element_data.update({
                    "type": "image",
                    "image_properties": shape.get("shapeProperties", {}).get("shapeFill", {}).get("pictureFill", {})
                })
        
        # Handle tables
        elif "table" in element:
            table_data = extract_table_structure(element["table"])
            element_data.update(table_data)
        
        # Handle images
        elif "image" in element:
            element_data.update({
                "type": "image",
                "image_properties": element.get("image", {})
            })
        
        # Handle videos
        elif "video" in element:
            element_data.update({
                "type": "video",
                "video_properties": element.get("video", {})
            })
        
        # Handle charts
        elif "sheetsChart" in element:
            element_data.update({
                "type": "chart",
                "chart_properties": element.get("sheetsChart", {})
            })
        
        else:
            element_data["type"] = "unknown"
        
        elements.append(element_data)
    
    return elements


def extract_slide_text(slide: dict) -> str:
    """Legacy function for plain text extraction."""
    texts: List[str] = []

    # Prefer title placeholder first if present
    title_candidates: List[str] = []
    for element in slide.get("pageElements", []):
        shape = element.get("shape")
        if not shape:
            continue
        placeholder = shape.get("placeholder")
        if placeholder and placeholder.get("type") == "TITLE":
            text_elements = shape.get("text", {}).get("textElements", [])
            title_text = extract_plain_text_from_paragraphs(extract_text_with_formatting(text_elements))
            if title_text:
                title_candidates.append(title_text)
    if title_candidates:
        texts.append("\n".join(title_candidates))

    # Then other shapes, tables, and images in order of appearance
    for element in slide.get("pageElements", []):
        if "shape" in element:
            text_elements = (
                element["shape"].get("text", {}).get("textElements", [])
            )
            body_text = extract_plain_text_from_paragraphs(extract_text_with_formatting(text_elements))
            if body_text:
                texts.append(body_text)
        elif "table" in element:
            table_text = extract_text_from_table(element["table"])
            if table_text:
                texts.append(table_text)
        elif "image" in element:
            texts.append("[image]")

    # Remove duplicates while preserving order
    seen = set()
    deduped: List[str] = []
    for t in texts:
        key = t
        if key and key not in seen:
            seen.add(key)
            deduped.append(t)

    combined = "\n".join([t for t in deduped if t]).strip()
    return combined


def extract_notes_text(slide: dict) -> str:
    notes_page = slide.get("notesPage") or slide.get("slideProperties", {}).get("notesPage", {})
    note_texts: List[str] = []
    for element in notes_page.get("pageElements", []):
        shape = element.get("shape")
        if not shape:
            continue
        text_elements = shape.get("text", {}).get("textElements", [])
        paragraphs = extract_text_with_formatting(text_elements)
        content = extract_plain_text_from_paragraphs(paragraphs)
        if content:
            note_texts.append(content)
    return "\n".join(note_texts).strip()


def fetch_presentation(service, presentation_id: str) -> dict:
    return (
        service.presentations().get(presentationId=presentation_id).execute()
    )


def extract_presentation_structure(presentation: dict) -> List[dict]:
    """Extract full presentation structure with formatting and positioning."""
    slides = presentation.get("slides", [])
    slide_data = []
    
    for slide in slides:
        slide_info = {
            "slide_id": slide.get("objectId"),
            "slide_index": slide.get("slideProperties", {}).get("index", 0),
            "layout": slide.get("slideProperties", {}).get("layout", "UNSPECIFIED"),
            "elements": extract_slide_elements(slide)
        }
        
        notes = extract_notes_text(slide)
        if notes:
            slide_info["speaker_notes"] = notes
        
        slide_data.append(slide_info)
    
    return slide_data


def write_slides_to_txt(
    out_dir: str,
    base_name: str,
    slide_texts: List[str],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = sanitize_filename(base_name) + ".txt"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(slide_texts, start=1):
            f.write(f"# Slide {idx}\n")
            if text:
                f.write(text)
            f.write("\n\n")

    return out_path


def load_conversion_rules():
    """Load conversion rules from rules.txt"""
    try:
        with open("rules.txt", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("rules.txt not found, using default formatting")
        return {}

def detect_content_type(text_content):
    """Detect the type of content to apply appropriate formatting"""
    if not text_content.strip():
        return "empty"
    
    # Check for section markers
    if "#" in text_content and len(text_content.split()) < 10:
        return "section_header"
    
    # Check for instructor info patterns
    if any(keyword in text_content.lower() for keyword in ["instructor", "email", "references"]):
        return "instructor_info"
    
    # Check for course header patterns
    if any(keyword in text_content for keyword in ["MSML", "Course", ":"]) and len(text_content.split()) < 8:
        return "course_header"
    
    # Check if it's mostly bullet points
    lines = text_content.strip().split('\n')
    bullet_lines = sum(1 for line in lines if line.strip().startswith(('*', '-', '•')))
    if bullet_lines > len(lines) * 0.6:
        return "bullet_slide"
    
    return "regular_content"

def format_text_with_rules(text, formatting_info=None):
    """Apply text formatting according to rules"""
    if not text:
        return ""
    
    # Handle LaTeX commands and special formatting
    text = text.replace('≈', '$\\approx$')
    text = text.replace('©', '\\copyright')
    text = text.replace('→', '$\\rightarrow$')
    text = text.replace('–', '--')
    
    # Apply bold/italic formatting if specified
    if formatting_info:
        if formatting_info.get('bold'):
            text = f"**{text}**"
        elif formatting_info.get('italic'):
            text = f"*{text}*"
    
    return text

def format_section_header(content, level="major"):
    """Format section headers according to rules"""
    content = content.strip().replace('#', '').strip()
    
    if level == "major":
        return f"# ##############################################################################\n# {content}\n# ##############################################################################\n\n* {content}"
    else:
        return f"## #############################################################################\n## {content}\n## #############################################################################\n\n* {content}"

def format_bullet_content(paragraphs, is_main_bullet_slide=False):
    """Format bullet point content according to rules"""
    output = []
    
    for para in paragraphs:
        para_content = ""
        for part in para.get("content_parts", []):
            content = part.get("content", "")
            if part.get("bold"):
                para_content += f"**{content}**"
            elif part.get("italic"):
                para_content += f"*{content}*"
            else:
                para_content += content
        
        para_content = para_content.rstrip('\n')
        if not para_content.strip():
            continue
        
        if para.get("bullet"):
            indent = "  " * para["bullet"].get("nestingLevel", 0)
            # Use - for normal bullets, and indent for nested
            bullet_char = "-"
            output.append(f"{indent}{bullet_char} {para_content}")
        else:
            output.append(para_content)
    
    return output

def write_slides_to_enhanced_txt(
    out_dir: str,
    base_name: str,
    slide_data: List[dict],
    image_map: dict,
    lecture_number: float = 1,
) -> str:
    """Write raw slide content for later processing by enhance_with_llm.py."""
    os.makedirs(out_dir, exist_ok=True)
    filename = sanitize_filename(base_name) + "_raw.txt"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, slide in enumerate(slide_data, start=1):
            f.write(f"=== SLIDE {idx} ===\n\n")
            
            for element in slide.get("elements", []):
                element_type = element.get("type", "unknown")

                if element_type == "text_shape":
                    is_title = element.get("placeholder_type") == "TITLE"
                    is_subtitle = element.get("placeholder_type") == "SUBTITLE"
                    
                    # Mark the type of text element
                    if is_title:
                        f.write("[TITLE]\n")
                    elif is_subtitle:
                        f.write("[SUBTITLE]\n")
                    else:
                        f.write("[TEXT]\n")
                    
                    # Extract raw text with minimal formatting info
                    paragraphs = element.get("formatted_content", [])
                    for para in paragraphs:
                        para_content = ""
                        for part in para.get("content_parts", []):
                            content = part.get("content", "")
                            # Keep basic formatting markers
                            if part.get("bold"):
                                para_content += f"[BOLD]{content}[/BOLD]"
                            elif part.get("italic"):
                                para_content += f"[ITALIC]{content}[/ITALIC]"
                            else:
                                para_content += content
                        
                        para_content = para_content.rstrip('\n')
                        if para_content.strip():
                            # Mark if it's a bullet point
                            if para.get("bullet"):
                                indent_level = para["bullet"].get("nestingLevel", 0)
                                f.write(f"[BULLET:level={indent_level}] {para_content}\n")
                            else:
                                f.write(f"{para_content}\n")
                    f.write("\n")
                
                elif element_type == "image":
                    object_id = element.get("object_id")
                    image_path = image_map.get(object_id, "image_not_found")
                    f.write(f"[IMAGE] {image_path}\n\n")

                elif element_type == "table":
                    f.write(f"[TABLE {element.get('rows', 0)}x{element.get('columns', 0)}]\n")
                    for cell in element.get("cells", []):
                        row = cell['row']
                        col = cell['col']
                        content = cell.get('content', '')
                        is_header = cell.get("is_header", False)
                        f.write(f"[CELL:{row},{col}{'|HEADER' if is_header else ''}] {content}\n")
                    f.write("\n")
                
                elif element_type == "video":
                    video_props = element.get("video_properties", {})
                    url = video_props.get('url', 'embedded video')
                    f.write(f"[VIDEO] {url}\n\n")
                
                elif element_type == "chart":
                    f.write("[CHART]\n\n")
            
            # Add speaker notes if present
            if slide.get("speaker_notes"):
                f.write(f"[SPEAKER_NOTES]\n{slide['speaker_notes']}\n\n")
            
            f.write("=== END SLIDE ===\n\n")

    return out_path


def process_one_link(
    slides_service,
    creds: "Credentials",
    url_or_id: str,
    out_dir: str,
    lecture_number: float = 1,
) -> Optional[str]:
    presentation_id = extract_presentation_id(url_or_id)
    if not presentation_id:
        logging.error("Could not parse presentation ID from: %s", url_or_id)
        return None

    logging.info("Fetching presentation: %s", presentation_id)
    try:
        presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
    except Exception as exc:
        logging.error("Failed to fetch presentation %s: %s", presentation_id, exc)
        return None

    title = presentation.get("title", presentation_id)
    
    slide_data = extract_presentation_structure(presentation)

    # Create images directory and download images in order
    lecture_safe = sanitize_lecture_number_for_filename(lecture_number)
    images_dir = os.path.join(out_dir, "images", f"lecture_{lecture_safe}")
    os.makedirs(images_dir, exist_ok=True)
    image_map = download_images_in_order(
        creds, slide_data, images_dir, lecture_number
    )

    try:
        out_path = write_slides_to_enhanced_txt(
            out_dir, title, slide_data, image_map, lecture_number
        )
    except Exception as exc:
        logging.error("Failed to write raw output for %s: %s", title, exc)
        return None

    logging.info("Wrote: %s", out_path)
    return out_path


def read_urls_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch text content from Google Slides presentations and save to `output/`."
            " By default, it reads URLs from `links.txt`."
        )
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    )
    parser.add_argument(
        "--lecture-start",
        type=float,  # Changed from int to float
        default=2,
        help="Starting lecture number (default: 2).",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        help="Google Slides links or IDs provided inline (overrides `links.txt`).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose or 1)

    if GOOGLE_IMPORT_ERROR is not None:
        logging.error(
            "Missing Google API libraries. Install with: "
            "pip install google-auth google-auth-oauthlib google-api-python-client"
        )
        return 2

    try:
        creds = authenticate_and_get_creds()
        slides_service = build("slides", "v1", credentials=creds, cache_discovery=False)
        # drive_service = build("drive", "v3", credentials=creds, cache_discovery=False) # No longer needed for image download
    except Exception as exc:
        logging.error("Authentication failed: %s", exc)
        return 2
    
    urls: List[str] = []
    # If URLs are passed as arguments, use them. Otherwise, look for links.txt.
    if args.urls:
        urls.extend(args.urls)
    elif os.path.exists("links.txt"):
        logging.info("No URLs provided via command line, reading from links.txt...")
        try:
            urls.extend(read_urls_from_file("links.txt"))
        except Exception as exc:
            logging.error("Failed to read links.txt: %s", exc)
            return 2
    
    if not urls:
        logging.warning(
            "No URLs to process. Provide them as arguments or in a `links.txt` file."
        )
        return 0

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    successes = 0
    # FIX: Handle float lecture_start properly
    for i, url in enumerate(urls):
        lecture_num = args.lecture_start + i  # Calculate the actual lecture number
        out_path = process_one_link(
            slides_service=slides_service,
            creds=creds,
            url_or_id=url,
            out_dir=out_dir,
            lecture_number=lecture_num,
        )
        if out_path:
            successes += 1

    logging.info("Completed. %d/%d presentations processed successfully.", successes, len(urls))
    return 0 if successes == len(urls) else 1


if __name__ == "__main__":
    sys.exit(main()) 