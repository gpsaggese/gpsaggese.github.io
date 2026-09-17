"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type RegexReplacement = string | ((substring: string, ...args: string[]) => string);
type RegexReplacementRule = [RegExp, RegexReplacement];

/**
 * Apply a set of regex replacements only to non-math segments of text.
 * Math blocks ($...$, $$...$$) are preserved untouched.
 */
function applyOutsideMath(
  text: string,
  replacements: RegexReplacementRule[],
): string {
  const parts = text.split(/(\$\$[\s\S]*?\$\$|\$(?:[^$\n]|\\\$)+?\$)/g);
  return parts
    .map((part) => {
      if (
        (part.startsWith("$$") && part.endsWith("$$")) ||
        (part.startsWith("$") && part.endsWith("$") && part.length > 2)
      ) {
        return part; // math block — skip
      }
      let f = part;
      for (const [re, repl] of replacements) {
        f = typeof repl === "string" ? f.replace(re, repl) : f.replace(re, repl);
      }
      return f;
    })
    .join("");
}

/**
 * Apply regex replacements only to math segments ($...$, $$...$$).
 * Inverse of applyOutsideMath — fixes broken commands INSIDE math blocks.
 */
function applyInsideMath(
  text: string,
  replacements: RegexReplacementRule[],
): string {
  const parts = text.split(/(\$\$[\s\S]*?\$\$|\$(?:[^$\n]|\\\$)+?\$)/g);
  return parts
    .map((part) => {
      const isMath =
        (part.startsWith("$$") && part.endsWith("$$")) ||
        (part.startsWith("$") && part.endsWith("$") && part.length > 2);
      if (!isMath) return part; // non-math — skip
      let f = part;
      for (const [re, repl] of replacements) {
        f = typeof repl === "string" ? f.replace(re, repl) : f.replace(re, repl);
      }
      return f;
    })
    .join("");
}

/**
 * Aggressively fix stray/broken LaTeX in LLM output before KaTeX processes it.
 */
function sanitizeLatex(text: string): string {
  let t = text;

  // === Pass -1: Un-double-escape backslashes ===
  // Some content arrives with \\text instead of \text (backend double-escaping).
  // In the browser string, this looks like two real backslashes before a command.
  // KaTeX interprets \\ as line-break, so \\text renders as linebreak + italic "text".
  // Fix: collapse \\ → \ before known LaTeX command names and spacing commands.
  t = t.replace(
    /\\\\(text(?:it|bf|rm|tt)?|math(?:rm|it|bf)|frac|sqrt|bar|hat|tilde|vec|overline|underline|times|approx|g?eq|leq|pm|Delta|Sigma|Omega|Gamma|Lambda|Pi|delta|sigma|omega|gamma|lambda|alpha|beta|eta|rho|chi|mu|theta|phi|psi|epsilon|kappa|tau|pi|zeta|nu|xi|infty|neq|sim|propto|cdot|ldots|ll|gg|equiv|[,;!>]|quad|qquad|hspace|vspace|left|right|big|Big|bigg|Bigg)(?![a-zA-Z])/g,
    "\\$1",
  );

  // === Pass 0: Fix markdown/GFM issues and corrupted Unicode ===
  // Fix unpaired ** bold markers. When content is split into paragraphs,
  // bold spans can break (opening ** in one paragraph, closing in another).
  // Count ** occurrences — if odd, strip all ** (better than broken rendering).
  const boldCount = (t.match(/\*\*/g) || []).length;
  if (boldCount % 2 !== 0) {
    t = t.replace(/\*\*/g, "");
  }
  t = applyOutsideMath(t, [
    // "~~1200" triggers GFM ~~strikethrough~~. Escape ~~ when not intentional.
    [/~~(?=\d|[A-Z])/g, "\\~\\~"],
    // Fix corrupted Unicode: Ö3b8 → θ, Ö3b7 → η, etc. (Python \u escape mangled)
    [/[Öö]3b8/g, "$\\theta$"],
    [/[Öö]3b7/g, "$\\eta$"],
    [/[Öö]3b1/g, "$\\alpha$"],
    [/[Öö]3b2/g, "$\\beta$"],
    [/[Öö]3b3/g, "$\\gamma$"],
    [/[Öö]3b4/g, "$\\delta$"],
    [/[Öö]3b5/g, "$\\epsilon$"],
    [/[Öö]3c3/g, "$\\sigma$"],
    [/[Öö]3c1/g, "$\\rho$"],
    [/[Öö]3c7/g, "$\\chi$"],
    [/[Öö]3bc/g, "$\\mu$"],
  ]);

  // === Pass 1: Compound patterns (must run first, before pieces get wrapped) ===
  t = applyOutsideMath(t, [
    // "text{foo}" → "$\text{foo}$" (Python \t ate backslash)
    [/(?<![\\a-zA-Z])text\{([^}]*)\}/g, (_m: string, inner: string) => `$\\text{${inner}}$`],
    [/(?<![\\a-zA-Z])textit\{([^}]*)\}/g, (_m: string, inner: string) => `$\\textit{${inner}}$`],
    [/(?<![\\a-zA-Z])textbf\{([^}]*)\}/g, (_m: string, inner: string) => `$\\textbf{${inner}}$`],
    [/(?<![\\a-zA-Z])mathrm\{([^}]*)\}/g, (_m: string, inner: string) => `$\\mathrm{${inner}}$`],
    [/(?<![\\a-zA-Z])mathit\{([^}]*)\}/g, (_m: string, inner: string) => `$\\mathit{${inner}}$`],
    // \text{foo} outside math (with backslash intact)
    [/\\(text|textit|textbf|textrm|texttt|mathrm|mathit|mathbf|bar|hat|tilde|vec|overline|underline|sqrt)\{([^}]*)\}/g,
      (_m: string, cmd: string, inner: string) => `$\\${cmd}{${inner}}$`],
  ]);

  // === Pass 2: Compound expressions (times10^{exp}, frac, etc.) ===
  t = applyOutsideMath(t, [
    // "times10^{-N}" or "times 10^{-N}" (backslash eaten)
    [/(?<![a-zA-Z])times\s*(\d+)\^\{([^}]*)\}/g,
      (_m: string, base: string, exp: string) => `$\\times ${base}^{${exp}}$`],
    [/(?<![a-zA-Z])times\s*(\d+)\^(-?\d+)/g,
      (_m: string, base: string, exp: string) => `$\\times ${base}^{${exp}}$`],
    // \frac{a}{b}
    [/\\frac\{([^}]*)\}\{([^}]*)\}/g,
      (_m: string, num: string, den: string) => `$\\frac{${num}}{${den}}$`],
  ]);

  // === Pass 3: Bare run-on commands (textmean, mathrmNaN, etc.) ===
  t = applyOutsideMath(t, [
    [/(?<![a-zA-Z])text([a-zA-Z_][a-zA-Z_0-9]*)/g,
      (_m: string, word: string) => `$\\text{${word.replace(/_/g, "\\_")}}$`],
    [/(?<![a-zA-Z])mathrm([a-zA-Z_][a-zA-Z_0-9]*)/g,
      (_m: string, word: string) => `$\\mathrm{${word}}$`],
    [/(?<![a-zA-Z])mathit([a-zA-Z_][a-zA-Z_0-9]*)/g,
      (_m: string, word: string) => `$\\mathit{${word}}$`],
    [/(?<![a-zA-Z])mathbf([a-zA-Z_][a-zA-Z_0-9]*)/g,
      (_m: string, word: string) => `$\\mathbf{${word}}$`],
  ]);

  // === Pass 4: Standalone symbols ===
  t = applyOutsideMath(t, [
    // \times, \approx etc. with backslash
    [/\\(times|approx|geq|leq|pm|Delta|delta|chi|rho|sigma|mu|alpha|beta|gamma|lambda|infty|neq|sim|propto|cdot|ldots|ge|le|ll|gg|equiv|eta|theta|phi|psi|omega|epsilon|kappa|tau|pi|zeta|nu|xi)(?![a-zA-Z{])/g,
      (_m: string, cmd: string) => `$\\${cmd}$`],
    // Bare Greek/math commands without backslash (Python ate them)
    [/(?<![a-zA-Z\\])(Delta|Sigma|Omega|Gamma|Lambda|eta|rho|chi|sigma|mu|alpha|beta|gamma|delta|lambda|theta|phi|psi|omega|epsilon|kappa|tau|pi|zeta|nu|xi|ll|gg|sim|approx|infty|neq|equiv|propto)(?=[^a-z]|$)/g,
      (_m: string, cmd: string) => `$\\${cmd}$`],
    // Bare "sim" and "approx" before numbers
    [/(?<![a-zA-Z])sim(\d)/g, (_m: string, d: string) => `$\\sim$${d}`],
    [/(?<![a-zA-Z])simC\(/g, () => `$\\sim$ C(`],
    [/(?<![a-zA-Z])approx(\d)/g, (_m: string, d: string) => `$\\approx$${d}`],
  ]);

  // === Pass 5: Exponents and subscripts ===
  t = applyOutsideMath(t, [
    // N^{exp}
    [/(\d+)\^\{(-?[\d.]+)\}/g,
      (_m: string, base: string, exp: string) => `$${base}^{${exp}}$`],
    // N^exp
    [/(\d+)\^(-?\d+)(?![\d}])/g,
      (_m: string, base: string, exp: string) => `$${base}^{${exp}}$`],
    // word_{sub}
    [/([a-zA-Z])\\?_\{([^}]*)\}/g,
      (_m: string, pre: string, sub: string) => `${pre}$_{${sub}}$`],
  ]);

  // === Pass 5b: Fix bare commands INSIDE math blocks ===
  // When LLM wraps content in $...$ but Python ate the backslash,
  // e.g. $DeltaP = textLV ActivePower(kW)$ → $\Delta P = \text{LV ActivePower(kW)}$
  t = applyInsideMath(t, [
    // Bare text/mathrm commands with braces: text{foo} → \text{foo}
    [/(?<!\\)text\{/g, "\\text{"],
    [/(?<!\\)textit\{/g, "\\textit{"],
    [/(?<!\\)textbf\{/g, "\\textbf{"],
    [/(?<!\\)mathrm\{/g, "\\mathrm{"],
    [/(?<!\\)mathit\{/g, "\\mathit{"],
    [/(?<!\\)mathbf\{/g, "\\mathbf{"],
    // Bare text run-on: textFoo or textMean(...) → \text{Foo} or \text{Mean}(...)
    [/(?<!\\)text([A-Z][a-zA-Z_]*)/g, (_m: string, word: string) => `\\text{${word}}`],
    // Bare times before digit: times10 → \times 10
    [/(?<!\\)times(\d)/g, (_m: string, d: string) => `\\times ${d}`],
    [/(?<!\\)times\s+(\d)/g, (_m: string, d: string) => `\\times ${d}`],
    // Bare Greek/math symbols without backslash
    [/(?<![a-zA-Z\\])(Delta|Sigma|Omega|Gamma|Lambda|Pi)(?=[^a-z]|$)/g,
      (_m: string, cmd: string) => `\\${cmd}`],
    [/(?<![a-zA-Z\\])(delta|sigma|omega|gamma|lambda|alpha|beta|eta|rho|chi|mu|theta|phi|psi|epsilon|kappa|tau|pi|zeta|nu|xi)(?=[^a-z]|$)/g,
      (_m: string, cmd: string) => `\\${cmd}`],
    // Bare times/approx/sim etc.
    [/(?<![a-zA-Z\\])(times|approx|geq|leq|pm|infty|neq|sim|propto|cdot|ldots|equiv)(?=[^a-z]|$)/g,
      (_m: string, cmd: string) => `\\${cmd}`],
    // Corrupted Unicode inside math: Ö0d7 → \times, ö3b8 → \theta, etc.
    [/[Öö]0d7/g, "\\times"],
    [/[Öö]3b8/g, "\\theta"],
    [/[Öö]3b7/g, "\\eta"],
    [/[Öö]3b1/g, "\\alpha"],
    [/[Öö]3b2/g, "\\beta"],
    [/[Öö]3b3/g, "\\gamma"],
    [/[Öö]3c3/g, "\\sigma"],
    [/[Öö]3c7/g, "\\chi"],
    [/[Öö]3bc/g, "\\mu"],
  ]);

  // === Pass 6: Merge adjacent math blocks ===
  // $\eta$^2 → $\eta^2$
  t = t.replace(/\$([^$]+)\$\^(\{[^}]+\}|\d+)/g, (_m, inner, exp) => `$${inner}^${exp}$`);
  // $\text{foo}$_{bar} → $\text{foo}_{bar}$
  t = t.replace(/\$([^$]+)\$_(\{[^}]+\})/g, (_m, inner, sub) => `$${inner}_${sub}$`);
  // $\Delta$R^2 → $\Delta R^2$ (merge symbol + adjacent letter with exponent)
  t = t.replace(/\$([^$]+)\$([A-Za-z])\^(\{[^}]+\}|\d+)/g,
    (_m, inner, letter, exp) => `$${inner} ${letter}^{${exp}}$`);
  // $\Delta$R → $\Delta R$ (merge symbol + adjacent single letter)
  t = t.replace(/\$([^$]+)\$([A-Za-z])(?=[\s=<>,;:)\]]|$)/g,
    (_m, inner, letter) => `$${inner} ${letter}$`);

  // Clean up empty math blocks
  t = t.replace(/\$\s*\$/g, "");

  return t;
}

interface StoryMarkdownProps {
  content: string;
  className?: string;
}

export default function StoryMarkdown({ content, className }: StoryMarkdownProps) {
  const sanitized = sanitizeLatex(content);
  return (
    <div className={`report-prose prose prose-sm max-w-none text-on-surface-variant ${className || ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {sanitized}
      </ReactMarkdown>
    </div>
  );
}

export { sanitizeLatex };
