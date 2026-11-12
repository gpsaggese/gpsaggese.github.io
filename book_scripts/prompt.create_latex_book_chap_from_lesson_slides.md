# Goal
- You are a college professor expert in AI, machine learning, and big data
- Convert lecture slides into a textbook chapter by expanding each slide's
  content into a clear, accessible explanations formatted for LaTeX

## Structural Hierarchy
- Convert slide hierarchy into proper document structure:
  - **H1 (top-level topic)** → Chapter (use `\chapter{Title}` command)
    - One chapter per main topic
  - **H2 (major section)** → Section (use `\section{Title}` in LaTeX)
    - Major subsection of chapter
  - **H3+ (minor sections)** → Subsection (use `\subsection{Title}` in LaTeX)
    - Further subdivisions as needed

- The active DIR is `/Users/saggese/src/umd_classes1`

## Output Format
- Generate LaTeX format with complete document structure
- Preserve slide organization with proper heading hierarchy
- Output file: Generate a LaTeX file in the sibling `book` directory than
  `lesson_source` (e.g., `msml610/book`) with the same filename as
  the input but using `.tex` extension

## Document Template - Springer SNmono Format
- Follow Springer Nature SNmono monograph class for compatibility with
  `$DIR/book_springer/book`
- Individual chapter files (standalone compilation during development):
  ```latex
  % Chapter file for [Chapter Title]
  % This chapter file can be compiled standalone or included in the root book.tex

  \chapter{[Chapter Title]}
  \label{chap:[unique-label]}

  \motto{Short motivational quote or key insight from the chapter (optional)}

  % Chapter content starts here
  ```

- Root file structure (book.tex) - only for reference:
  ```latex
  \documentclass[graybox,envcountchap,sectrefs]{SNmono}
  \usepackage{type1cm}
  \usepackage{graphicx}
  \usepackage{makeidx}
  \usepackage{newtxtext}
  \usepackage[varvw]{newtxmath}

  \begin{document}
  \frontmatter
  \include{dedication}
  \include{preface}
  \tableofcontents

  \mainmatter
  \include{chapter1}
  \include{chapter2}

  \backmatter
  \include{appendix}
  \include{glossary}
  \end{document}
  ```

## Add a Comment to Track the Version
- Get the `<GIT_HASH>` of the current version of the Git repo with:
  ```
  > git rev-parse --short HEAD
  ```
- Get the `<TIMESTAMP>` with:
  ```
  > timestamp
  ```
- Add a comment at the beginning of the file to track when the document was
  generated
  ```latex
  % git_hash=<GIT_HASH>, timestamp=<TIMESTAMP>
  ```

## Writing Style
- **Audience**:
  - Undergraduate or early graduate students
  - Assume familiarity with foundational concepts in AI, machine learning,
    statistics, computer science
  - Senior ML engineers and data scientists with a statistics and probabilistic
    ML background who build production decision systems
  - Working knowledge of causal basics (DAGs, SCMs, do-calculus) assumed

- **Tone**: Clear, conversational, academic but not dense
- **Language**: Use precise technical terms, avoid jargon or overly fancy synonyms
- **No AI slop**: Follow the directions `.claude/skills/text.humanize/SKILL.md`
  to avoid AI slop and not-human sounding text
- **Word count per slide**: 350–400 words (adjust as needed for content depth)

## Content Guidelines
- **Explain concepts**: Define unfamiliar terms; provide intuition before formalism
- **Add context**: Connect slide topics to real-world applications or broader themes
- **Highlight key points**:
  - Use `\emph{text}` for highlighting key terms, concepts, and definitions
  - Use `\textit{text}` for italics sparingly (emphasis only, not decoration)
  - Use `\texttt{text}` for code, file names, or technical identifiers
  - Use `\textbf{text}` for section headers and structural elements only
- **Algorithms & Pseudocode**: Use `\begin{algorithm}...\end{algorithm}` environment for structured
  algorithms, pseudocode, or procedural content
  - Format with proper indentation using spacing
  - Use `\textbf{keyword}` for language keywords (function, if, loop, return, etc.) — these are structural, not content highlighting
  - Use symbolic notation ($\sigma$, $\notin$, etc.) where appropriate
  - Alternative: use `\begin{programcode}{Title}` environment for code blocks
- **Formulas**: Preserve all mathematical formulas
  - Inline math: use `$formula$` for text-like expressions (e.g., `$f(n) = g(n) + h(n)$`)
  - Display math: use `\begin{equation}...\end{equation}` or `\begin{align}...\end{align}` for
    important equations (auto-numbered)
  - Use `\text{...}` inside math mode for text, NOT `\vbox` or `\hbox`
  - Example: `\begin{equation}\label{eq:main} f(n) = g(n) + h(n) \end{equation}`
- **Special Environments** (Springer SNmono):
  - Use `\begin{theorem}...\end{theorem}` for theoretical results
  - Use `\begin{definition}...\end{definition}` for definitions
  - Use `\begin{proof}...\end{proof}` for proofs
  - Use `\begin{svgraybox}...\end{svgraybox}` for emphasized paragraphs (15% gray background)
  - Use `\begin{important}{Title}...\end{important}` for important notes
  - Use `\begin{warning}{Title}...\end{warning}` for warnings/cautions
  - Use `\begin{tips}{Tips}...\end{tips}` for helpful hints

## Slide Source Markers
- Annotate the generated chapter file with pointers back to the exact heading
  in the source `lectures_source/*.txt` file that each LaTeX construct was
  generated from
- Before each `\section`, `\subsection`, or `\textbf{...}` run-in heading that
  corresponds to a slide heading, add a one-line comment reproducing the
  original heading text with its markdown prefix (`#`, `##`, or `*`) exactly
  as it appears in the source `.txt` file, plus the source file and line
  number (`%` is the comment marker here since the output is LaTeX, not
  Markdown):
  ```latex
  % From: <file>:<line num> '<prefix> <heading text>'
  ```
  - `<file>` is a repo-root-relative path to the `lectures_source/*.txt` file
    being converted, `<line num>` its line
  - See `.claude/skills/slides.rules.md` ("Source Attribution") for the full
    `// From: <file>:<line num> '<title>'` convention
- Mapping from source heading level to LaTeX construct:
  - H1 (`#`) in the source maps to `\section{...}`:
    ```
    # Why Decisions, Not Predictions
    ```
    becomes
    ```latex
    % From: msml610/lectures_source/Lesson01.1-Why_Decisions.txt:12 '# Why Decisions, Not Predictions'
    \section{Why Decisions, Not Predictions}
    ```
  - H2 (`##`) in the source maps to `\subsection{...}`:
    ```
    ## When Correlation Misleads
    ```
    becomes
    ```latex
    % From: msml610/lectures_source/Lesson01.1-Why_Decisions.txt:34 '## When Correlation Misleads'
    \subsection{When Correlation Misleads}
    ```
  - Slide-level (`*`) heading in the source maps to a `\textbf{...}` run-in
    heading with em-dash continuation when followed by body text:
    ```
    * Why Traditional ML Falls Short
    Traditional machine learning was built to answer one question well...
    ```
    becomes
    ```latex
    % From: msml610/lectures_source/Lesson01.1-Why_Decisions.txt:58 '* Why Traditional ML Falls Short'
    \textbf{Why Traditional ML Falls Short} --- Traditional machine learning was
    built to answer one question well...
    ```
    (Note: Use em-dash ` --- ` to connect heading to text; omit dash if heading is followed by structure like `\begin{definition}` or `\begin{itemize}`)

## Use Lists
- Use lists when possible
  - E.g.,
    ```
    - Use **item1** when:
      - ...

    - Use **item2** when:
      - ...

    - Use **item3** when:
      - ...
    ```
    becomes
    ```latex
    \begin{itemize}
      \item Use \textbf{item1} when:
        \begin{itemize}
          \item ...
        \end{itemize}

      \item Use \textbf{item2} when:
        \begin{itemize}
          \item ...
        \end{itemize}

      \item Use \textbf{item3} when:
        \begin{itemize}
          \item ...
        \end{itemize}
    \end{itemize}
    ```

## Figure Identification Process (Springer SNmono)
1. Scan source material for explicitly referenced image files
   - Look for: `\includegraphics{...}`, `![](...)`, file paths ending in `.png`,
     `.jpg`, `.svg`, `.eps`
   - Springer recommends .eps or .pdf for vector graphics, .png for raster
   - Use the following format:
     ```latex
     \begin{figure}[!t]
       \centering
       \includegraphics[width=7cm]{path/to/figure.png}
       \caption{Concise description of figure content and relevance.}
       \label{fig:chart}
     \end{figure}
     ```
   - For narrower figures (< 7.8 cm), use `\sidecaption`:
     ```latex
     \begin{figure}
       \sidecaption
       \includegraphics[width=6cm]{path/to/figure.png}
       \caption{Figure caption for side placement.}
       \label{fig:sidebar}
     \end{figure}
     ```
   - Each figure requirements:
     - Must have label `\label{fig:<description>}`
     - One-line caption describing content
     - Referenced in text using `\ref{fig:diagram}`, `\autoref{fig:diagram}`, or `Fig.~\ref{fig:diagram}`
     - Placed close to where it's referred to
     - Width specified in cm (typical: 5-9 cm for Springer layout)
   - Commented figure metadata:
     ```
     % rendered_image:begin
     % filename: path/to/figure.png
     % label: fig:my_label
     % caption: This is a caption
     %          that can span multiple lines
     % rendered_image:end
     ```

2. Document each figure with:
   - Filename and location
   - Brief description of what it shows
   - Recommended section placement in the chapter
3. For diagrams in code blocks (graphviz, mermaid):
   - Extract the code
   - Save to figures directory
   - Embed in appropriate section with caption

## Best Practices (Springer SNmono)
- Every heading should be followed by at least a short passage of text (avoid heading immediately after heading)
- First line after a heading is NOT indented; subsequent paragraph first lines ARE indented
- Use the algorithm environment for structured procedures and pseudocode
- Keep definitions concise and clear
- Use `\begin{enumerate}...\end{enumerate}` for numbered lists (auto-formatted by Springer style)
- Use `\begin{itemize}...\end{itemize}` for bullet lists
- Use `\begin{description}[Type 1]...\end{description}` for term/definition lists
- Include illustrative examples alongside abstract concepts
- Use `\clearpage` strategically between major sections (or `\pagebreak`)
- Maintain consistent emphasis: italic (`\emph{}`) for key terms and concepts, italics (`\textit{}`) for general emphasis, bold (`\textbf{}`) for structural elements only
- Present formulas clearly with both intuition and formal notation
- Always label important sections/chapters with unique `\label{}` commands for cross-references
- Use `Sect.~\ref{sec:label}`, `Fig.~\ref{fig:label}`, `Table~\ref{tab:label}`, `Eq.~\eqref{eq:label}` for references
- Use footnotes judiciously; they will be placed at page bottom
- Avoid adding custom packages; use only what SNmono provides

## Constraints
- Preserve the complete logical flow and organization of the original slides
- Do not add examples not present in the source material unless they directly
  illustrate a concept from the slides
- Do not invent or extrapolate beyond the slide content
- Maintain all mathematical formulas, notation, and symbolic expressions exactly
- Use only components (chapters, sections, algorithm boxes, figures)

## LaTeX Syntax Requirements (Springer SNmono)
- Follow LaTeX (not markdown or typst) syntax strictly:
  - Use `\textbf{text}` for bold, NOT `**text**`
  - Use `\textit{text}` for italic emphasis, NOT `*text*` (though * works in some contexts)
  - Use `\texttt{text}` for typewriter/code font
  - Use `\textsl{text}` for slanted text (rarely)
  - Use `\emph{text}` for emphasis (context-aware, usually italic)
- Heading structure (auto-numbered unless marked with *):
  - `\chapter{Title}` - chapter level (in chapter file)
  - `\section{Heading}` - section level
  - `\subsection{Subheading}` - subsection level
  - `\subsubsection{...}` - further subdivisions
  - `\paragraph{Head}` - run-in heading (no new line)
  - `\subparagraph{Head}` - smaller run-in heading
- For formulas: preserve all mathematical expressions
  - Inline math: `$notation$` for LaTeX math mode
  - Display math: `\begin{equation}...\end{equation}` (numbered) or `\begin{align}...\end{align}` (multi-line)
  - Use `\text{...}` for text within math mode, NOT `\vbox` or `\hbox`
  - Use `\label{eq:name}` for equation references, `\eqref{eq:name}` to reference
- Use `\begin{algorithm}...\end{algorithm}` or `\begin{programcode}{Title}...\end{programcode}`
  - Format pseudocode with `\textbf{keyword}` for language keywords
  - Use proper indentation with spacing/tabs
- Use `\begin{figure}[!t]...\includegraphics{...}...\end{figure}` for images
  - Image paths relative to root book directory (use `../` as needed from chapter location)
  - Supported formats: .eps (preferred), .pdf, .png, .jpg
  - Avoid .svg (may not render in all formats); convert to .eps or .pdf if needed
- Use `\begin{table}[!t]...\end{table}` with `\begin{tabular}...\end{tabular}` for tables
- Use `\cite{label}` for citations; avoid linking from bibliography back to text
- Do NOT use: `\xdef`, `\fancyhdr`, `\awide`, `\enumerate`, `\eqnitem` (reserved)
- Do NOT use packages: `tikz`, `xy`, `pstricks`, `color` (use `xcolor` if needed with CMYK)
- Do NOT use `\pagereference` or `\pagebreak` with fixed page sizes
- Do NOT use `\vfill`, `\vspace{...}` excessively
- Do NOT nest complex structures in `multicol` or `minipage`
- Avoid `\def` and `\newcommand` for new definitions; use predefined SNmono commands
- Always close `\textbf{`, `\textit{}`, `\begin{...}` with matching closing syntax

## Springer SNmono Specific Requirements
- **Document Class**: Must use `\documentclass{SNmono}` when compiling in root context
- **Chapter Labels**: Always include `\label{chap:unique-identifier}` after `\chapter{...}`
- **Section Labels**: Use `\label{sec:N}` for sections, `\label{subsec:N}` for subsections
- **Theorem Numbering**: Controlled by class options (`envcountchap` for chapter-wise numbering)
- **Motto**: Optional `\motto{...}` command for motivational quote at chapter start
- **Footnotes**: Automatically placed at page bottom (Springer default)
- **Monograph vs. Edited Book**: Generated chapters are for monograph use (single author/editor per chapter)
- **Index Entries**: Use `\index{term}` for terms to appear in book index (if enabled)
- **Vector Notation**: Class option `vecphys` available for physics texts (vectors as bold italic)
- **Graybox Option**: Class option `graybox` enables `\begin{svgraybox}...\end{svgraybox}` (15% gray background)
- **Color Usage**: Only CMYK colors supported for print; avoid RGB; use `\xcolor{cmyk}` if needed
- **Fonts**: Springer uses Times/Times New Roman; class handles automatically; avoid custom font commands
- **Hyperlinks**: Hyperref included; links should point to book content only, not external web
- **Accessibility**: Ensure alt-text equivalent (caption) for all figures
- **Line Length**: Respect Springer's line length (~70-80 characters at default font size)
- **Widow/Orphan Control**: Springer handles automatically; do not force with `\widowpenalty` or `\clubpenalty`

## Verification Step
- After generating the chapter file:
  1. For standalone chapter verification, compile in isolation:
     ```bash
     > cd $DIR/book_springer/book
     > pdflatex [filename.tex]
     ```
  2. For integration with root book:
     - Add `\include{filename-without-.tex}` to book.tex in appropriate section
     - Compile root file: `pdflatex book.tex`
     - May need to run twice for cross-references and TOC to update
  3. File MUST compile without errors
  4. If compilation fails:
     - Check image paths are relative to root book directory (from `$DIR/book_springer/book/`)
     - Verify all `\textbf{`, `\textit{}`, `\begin{...}` are properly closed with `\end{...}`
     - Verify all `\label{}` and `\ref{}` commands are paired correctly
     - Ensure all formulas use valid LaTeX syntax (no `#` or `==` from other formats)
     - Check that `\cite{}` commands reference valid keys if using bibliography
     - Fix errors and retry compilation
  5. Check for warnings:
     - Underfull/overfull `\hbox` warnings (adjust spacing if excessive)
     - Undefined references (ensure all `\label{}`/`\ref{}` pairs match)
     - Float placement warnings (figures/tables moved from intended location)
  6. Report success only when compilation is clean with no errors
  7. For final validation in full book context:
     ```bash
     > cd $DIR/book_springer/book
     > pdflatex book.tex
     > pdflatex book.tex  # Run twice for references
     ```
