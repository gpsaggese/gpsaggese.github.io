/**
 * Compare baseline and current cell outputs to detect changes.
 * Used to determine if a story needs regeneration after notebook edits.
 */
import type { CellOutput } from "@/lib/types";

export interface OutputDiff {
  metric: string;
  oldValue: string;
  newValue: string;
  changeType: "numeric" | "text" | "plot";
}

function extractNumbers(text: string): Map<string, number> {
  const nums = new Map<string, number>();
  const re = /(\w[\w\s]*?)[:=]\s*(-?\d+\.?\d*)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    nums.set(m[1].trim(), parseFloat(m[2]));
  }
  return nums;
}

function outputToText(outputs: CellOutput[]): string {
  return outputs
    .map((o) => {
      if (o.text) return o.text;
      if (o.data?.["text/plain"]) return o.data["text/plain"];
      return "";
    })
    .join("\n");
}

function hashString(s: string): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    const chr = s.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash |= 0;
  }
  return hash.toString(16);
}

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return JSON.stringify(value);
}

export function compare(
  baseline: CellOutput[],
  current: CellOutput[]
): OutputDiff[] {
  const diffs: OutputDiff[] = [];

  const baseText = outputToText(baseline);
  const currText = outputToText(current);

  // Numeric comparison
  const baseNums = extractNumbers(baseText);
  const currNums = extractNumbers(currText);

  for (const [key, baseVal] of baseNums) {
    const currVal = currNums.get(key);
    if (currVal !== undefined && baseVal !== 0) {
      const relChange = Math.abs((currVal - baseVal) / baseVal);
      if (relChange > 0.05) {
        diffs.push({
          metric: key,
          oldValue: baseVal.toString(),
          newValue: currVal.toString(),
          changeType: "numeric",
        });
      }
    }
  }

  // Text comparison (if no numeric diffs found and text changed significantly)
  if (diffs.length === 0 && baseText !== currText && baseText.length > 0) {
    const lenChange =
      Math.abs(baseText.length - currText.length) / Math.max(baseText.length, 1);
    if (lenChange > 0.2 || hashString(baseText) !== hashString(currText)) {
      diffs.push({
        metric: "output_text",
        oldValue: baseText.slice(0, 100),
        newValue: currText.slice(0, 100),
        changeType: "text",
      });
    }
  }

  // Plot comparison (check if base64 images changed)
  const basePlots = baseline
    .filter((o) => o.data?.["image/png"])
    .map((o) => asText(o.data?.["image/png"]));
  const currPlots = current
    .filter((o) => o.data?.["image/png"])
    .map((o) => asText(o.data?.["image/png"]));

  if (basePlots.length > 0 || currPlots.length > 0) {
    const baseHashes = basePlots.map((p) => hashString(p));
    const currHashes = currPlots.map((p) => hashString(p));

    if (JSON.stringify(baseHashes) !== JSON.stringify(currHashes)) {
      diffs.push({
        metric: "plot_output",
        oldValue: `${basePlots.length} plot(s)`,
        newValue: `${currPlots.length} plot(s)`,
        changeType: "plot",
      });
    }
  }

  return diffs;
}
