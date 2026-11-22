import { create } from "zustand";
import type { StorySection, StoryTakeaway } from "@/lib/types";

function normalizeWhitespace(text: string): string {
  return text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function splitSummaryBlocks(text: string): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return [];

  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (paragraphs.length >= 2) return paragraphs;

  const sentences = normalized
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (sentences.length <= 3) return sentences.length ? sentences : [normalized];

  const chunkSize = Math.ceil(sentences.length / 3);
  const chunks: string[] = [];
  for (let i = 0; i < sentences.length; i += chunkSize) {
    chunks.push(sentences.slice(i, i + chunkSize).join(" "));
  }
  return chunks.filter(Boolean);
}

function titleCase(label: string): string {
  return label
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function clipPhrase(text: string, maxWords = 8): string {
  const words = text
    .replace(/[*_`>#-]+/g, "")
    .split(/\s+/)
    .filter(Boolean);
  return words.slice(0, maxWords).join(" ");
}

function stripLeadingFiller(text: string): string {
  return text
    .replace(/^(?:the|this|that|these|those|a|an|our|their)\s+/i, "")
    .replace(/^(?:key|main|primary|important)\s+/i, "")
    .replace(/^(?:dataset|analysis|report|summary|finding|findings|result|results)\s+/i, "")
    .trim();
}

function splitIntoSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+(?=[A-Z0-9(])/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function splitTakeawayPoints(text: string): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return [];

  const bulletLines = normalized
    .split(/\n+/)
    .map((part) => part.trim())
    .filter((part) => /^[•*-]\s+/.test(part));

  if (bulletLines.length > 0) {
    return bulletLines
      .map((point) => point.replace(/^[•*-]\s*/, "").trim())
      .filter(Boolean)
      .map((point) => point.replace(/\s+/g, " ").trim())
      .slice(0, 4);
  }

  const sentences = splitIntoSentences(normalized.replace(/\n+/g, " "));
  const points =
    sentences.length > 1
      ? sentences
      : normalized
          .split(/(?:\n|;\s+|:\s+)/)
          .map((part) => part.trim())
          .filter(Boolean);

  return points
    .map((point) => point.replace(/^[•*-]\s*/, "").trim())
    .filter(Boolean)
    .slice(0, 4)
    .map((point) => point.replace(/\s+/g, " ").trim());
}

function deriveTakeawayTitle(text: string, index: number): string {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return `Takeaway ${index + 1}`;

  const lower = normalized.toLowerCase();
  const semanticTitles: Array<[RegExp, string]> = [
    [/\b(correlat|assoc|relationship|co-?movement|multicollinear)\b/i, "Correlation Signal"],
    [/\b(outlier|anomal|exception|deviation|residual)\b/i, "Outlier Pattern"],
    [/\b(missing|null|gap|hole|coverage)\b/i, "Missingness Pattern"],
    [/\b(trend|rise|fall|increase|decrease|decline)\b/i, "Trend Signal"],
    [/\b(distribution|spread|skew|tail|histogram|ecdf)\b/i, "Distribution Shape"],
    [/\b(power|energy|wind|speed|direction)\b/i, "Wind-Power Pattern"],
    [/\b(hypothesis|test|p-value|significant|confirmed|refuted)\b/i, "Hypothesis Check"],
    [/\b(feature|column|series|dataset|rows|observations)\b/i, "Dataset Profile"],
  ];

  for (const [pattern, title] of semanticTitles) {
    if (pattern.test(lower)) return title;
  }

  const candidateWords = normalized
    .replace(/[^A-Za-z0-9\s-]/g, " ")
    .split(/\s+/)
    .map((word) => word.trim())
    .filter(Boolean)
    .filter((word) => word.length > 3)
    .filter(
      (word) =>
        !/^(?:with|from|that|this|these|those|have|has|had|been|were|was|are|and|or|but|then|when|where|what|which|into|over|under|than|the|for|in|on|of|to|by|as|at|an|a|analysis|summary|result|results|finding|findings|takeaway|takeaways)$/i.test(
          word
        )
    );

  const clipped = titleCase(stripLeadingFiller(candidateWords.slice(0, 4).join(" ")));
  if (clipped) return clipped;

  const fallback = titleCase(
    clipPhrase(
      stripLeadingFiller(
        normalized
          .split(/[,;:]/)[0]
          .replace(/^[•*-]\s*/, "")
          .replace(/\b(hypothesis|analysis|findings|results|shows|showed|indicates|suggests|demonstrates|reveals|contains|includes|with|and)\b/gi, "")
          .replace(/\s+/g, " ")
          .trim()
      ),
      4
    )
  );

  return fallback || `Takeaway ${index + 1}`;
}

function iconForTakeaway(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes("correlat")) return "hub";
  if (lower.includes("outlier") || lower.includes("anomal")) return "warning";
  if (lower.includes("trend") || lower.includes("increase") || lower.includes("decline")) return "trending_up";
  if (lower.includes("distribution") || lower.includes("spread")) return "bar_chart";
  if (lower.includes("missing") || lower.includes("null")) return "report";
  if (lower.includes("wind") || lower.includes("speed")) return "air";
  if (lower.includes("power") || lower.includes("energy")) return "electric_bolt";
  if (lower.includes("next step") || lower.includes("recommend")) return "task_alt";
  return "insights";
}

function buildTakeaways(summary: string): StoryTakeaway[] {
  const blocks = splitSummaryBlocks(summary);
  return blocks.slice(0, 3).map((body, index) => {
    const title = deriveTakeawayTitle(body, index);
    const points = splitTakeawayPoints(body);
    return {
      id: `takeaway-${index + 1}`,
      title,
      body,
      points: points.length > 0 ? points : [body],
      icon: iconForTakeaway(body),
      index: index + 1,
    };
  });
}

interface StoryState {
  title: string;
  executiveSummary: string;
  sections: StorySection[];
  generatedAt: string;
  summaryBlocks: string[];
  takeaways: StoryTakeaway[];
  loading: boolean;
  error: string | null;
  setStory: (story: {
    title: string;
    executive_summary: string;
    sections: StorySection[];
    generated_at: string;
  }) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clear: () => void;
}

export const useStoryStore = create<StoryState>((set) => ({
  title: "",
  executiveSummary: "",
  sections: [],
  generatedAt: "",
  summaryBlocks: [],
  takeaways: [],
  loading: false,
  error: null,
  setStory: (story) =>
    set({
      title: story.title,
      executiveSummary: normalizeWhitespace(story.executive_summary),
      sections: story.sections,
      generatedAt: story.generated_at,
      summaryBlocks: splitSummaryBlocks(story.executive_summary),
      takeaways: buildTakeaways(story.executive_summary),
      loading: false,
      error: null,
    }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),
  clear: () =>
    set({
      title: "",
      executiveSummary: "",
      sections: [],
      generatedAt: "",
      summaryBlocks: [],
      takeaways: [],
      loading: false,
      error: null,
    }),
}));
