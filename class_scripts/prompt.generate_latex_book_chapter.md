# LaTeX Book Chapter: Springer

Mode-specific instructions for `--mode springer_latex`. See
`prompt.generate_book_chapter_common.md` for the shared style guide
(audience, tone, content rules, constraints): this file covers only the
Springer SNmono LaTeX syntax and structure.

Copied and adapted from
`/Users/saggese/src/notes1/book_proposals/prompt.create_latex_book_chap_from_lesson_slides.md`.

## Output Format

- Generate LaTeX with a complete document structure
- Do not include `\documentclass`, `\begin{document}`, or a preamble: emit
  only a standalone chapter file (see Document Template below), since it is
  meant to be `\include`d in a root `book.tex`

## Document Template

```latex
%%%% Chapter file for [Chapter Title] %%%%
% This chapter file can be compiled standalone or included in the root book.tex

\chapter{[Chapter Title]}
\label{chap:[unique-label]}

\motto{Short motivational quote or key insight from the chapter (optional)}

% Chapter content starts here
```

## Structural Hierarchy → LaTeX

- H1 (`#`) → `\section{Title}`
- H2 (`##`) → `\subsection{Title}`
- H3+ (`###`+) → `\subsubsection{Title}`
- Slide-level heading (`*`) → `\textbf{Heading}` run-in heading with an
  em-dash continuation when followed by body text:
  ```
  * Why Traditional ML Falls Short
  Traditional machine learning was built to answer one question well...
  ```
  becomes
  ```latex
  \textbf{Why Traditional ML Falls Short} --- Traditional machine learning was
  built to answer one question well...
  ```
  (Omit the dash if the heading is followed by structure like
  `\begin{definition}` or `\begin{itemize}`.)

## Source Attribution

- Comment marker: `%`
  ```latex
  % From: msml610/lectures_source/Lesson01.1-Why_Decisions.smd:12 '# Why Decisions, Not Predictions'
  \section{Why Decisions, Not Predictions}
  ```

## Highlighting and Emphasis

- Use `\emph{text}` for highlighting key terms, concepts, and definitions
- Use `\textit{text}` for italics sparingly (emphasis only, not decoration)
- Use `\texttt{text}` for code, file names, or technical identifiers
- Use `\textbf{text}` for section headers and structural elements only

## Algorithms and Pseudocode

- Use `\begin{algorithm}...\end{algorithm}` for structured algorithms,
  pseudocode, or procedural content
  - Use `\textbf{keyword}` for language keywords (function, if, loop,
    return, etc.): these are structural, not content highlighting
  - Use symbolic notation ($\sigma$, $\notin$, etc.) where appropriate
- Alternative: use `\begin{programcode}{Title}...\end{programcode}` for code
  blocks

## Formulas

- Inline math: `$formula$` for text-like expressions (e.g.,
  `$f(n) = g(n) + h(n)$`)
- Display math: `\begin{equation}...\end{equation}` or
  `\begin{align}...\end{align}` for important equations (auto-numbered)
- Use `\text{...}` inside math mode for text, NOT `\vbox` or `\hbox`
- Example: `\begin{equation}\label{eq:main} f(n) = g(n) + h(n) \end{equation}`

## Special Environments (Springer SNmono)

- `\begin{theorem}...\end{theorem}` for theoretical results
- `\begin{definition}...\end{definition}` for definitions
- `\begin{proof}...\end{proof}` for proofs
- `\begin{svgraybox}...\end{svgraybox}` for emphasized paragraphs (15% gray
  background)
- `\begin{important}{Title}...\end{important}` for important notes
- `\begin{warning}{Title}...\end{warning}` for warnings/cautions
- `\begin{tips}{Tips}...\end{tips}` for helpful hints

## Use Lists

```latex
\begin{itemize}
  \item Use \textbf{item1} when:
    \begin{itemize}
      \item ...
    \end{itemize}
\end{itemize}
```

- Use `\begin{enumerate}...\end{enumerate}` for numbered lists
- Use `\begin{description}[Type 1]...\end{description}` for term/definition
  lists

## Figures

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
- Every figure must have `\label{fig:<description>}`, a one-line caption,
  and be referenced in the text via `\ref{fig:diagram}`,
  `\autoref{fig:diagram}`, or `Fig.~\ref{fig:diagram}`
- Width in cm (typical: 5-9 cm for Springer layout); image paths are
  relative to the root book directory

## LaTeX Syntax Requirements (Springer SNmono)

- Follow LaTeX (not markdown or typst) syntax strictly: `\textbf{}` not
  `**text**`, `\textit{}` not `*text*`, `\texttt{}` for code
- Heading structure (auto-numbered unless marked with `*`):
  `\section{}`, `\subsection{}`, `\subsubsection{}`, `\paragraph{}` (run-in,
  no new line), `\subparagraph{}` (smaller run-in)
- Use `\cite{label}` for citations; avoid linking from bibliography back to
  text
- Do NOT use: `\xdef`, `\fancyhdr`, `\awide`, `\enumerate`, `\eqnitem`
  (reserved)
- Do NOT use packages: `tikz`, `xy`, `pstricks`, `color` (use `xcolor` with
  CMYK if needed)
- Do NOT use `\vfill`, `\vspace{...}` excessively; avoid `\def` and
  `\newcommand`, use predefined SNmono commands instead
- Always close `\textbf{`, `\textit{}`, `\begin{...}` with matching closing
  syntax

## Springer SNmono Specific Requirements

- Always include `\label{chap:unique-identifier}` after `\chapter{...}`
- Use `\label{sec:N}` for sections, `\label{subsec:N}` for subsections
- `\motto{...}` is optional, for a motivational quote at chapter start
- Only CMYK colors are supported for print; avoid RGB
- Use `\index{term}` for terms that should appear in the book index
- Ensure alt-text equivalent (caption) for every figure
- Use `Sect.~\ref{sec:label}`, `Fig.~\ref{fig:label}`,
  `Table~\ref{tab:label}`, `Eq.~\eqref{eq:label}` for cross-references
