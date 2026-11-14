// Pandoc template for the official Typst "charged-ieee" IEEE conference
// template (https://github.com/typst/templates/tree/main/charged-ieee).
//
// Usage (see ../Makefile):
//   pandoc paper.md \
//     --from markdown+smart \
//     --template templates/ieee-template.typ \
//     --to typst \
//     --pdf-engine typst \
//     --resource-path figures \
//     -o paper.pdf
//
// Pandoc metadata (YAML front matter in paper.md) is mapped to the named
// arguments of the `ieee()` function from the charged-ieee package. Only
// the `bibliography` field is handled natively by Typst (not by pandoc's
// citeproc): citation keys in the Markdown (`[@key]` / `@key`) are passed
// through untouched by pandoc and resolved by Typst against
// `$bibliography$` using Typst's built-in "ieee" citation style, which the
// charged-ieee package already selects internally.
#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [$title$],
$if(author)$
  authors: (
$for(author)$
    (
      name: "$author.name$",
$if(author.department)$
      department: [$author.department$],
$endif$
$if(author.organization)$
      organization: [$author.organization$],
$endif$
$if(author.location)$
      location: [$author.location$],
$endif$
$if(author.email)$
      email: "$author.email$",
$endif$
    ),
$endfor$
  ),
$endif$
$if(abstract)$
  abstract: [$abstract$],
$endif$
$if(keywords)$
  index-terms: ($for(keywords)$[$keywords$]$sep$, $endfor$),
$endif$
$if(papersize)$
  paper-size: "$papersize$",
$endif$
$if(bibliography)$
  bibliography: bibliography($for(bibliography)$"$bibliography$"$sep$, $endfor$),
$endif$
  figure-supplement: [Fig.],
)

// Keep a unit and a section label on one line: no break at the slash of
// "m/s", at the hyphen of a label such as "VII-A", or inside "Section
// VII-A" (no hyphenation of "Section" either). The space before "m/s" and
// after "Section" is a non-breaking space in the Markdown source.
#show heading: set text(hyphenate: false)
#show "m/s": box
#show regex("(Sections?|Appendix)\u{a0}([IVX]+(-[A-H])?|[AB])"): box
#show regex("\\b[IVX]+-[A-H]\\b"): box

$body$
