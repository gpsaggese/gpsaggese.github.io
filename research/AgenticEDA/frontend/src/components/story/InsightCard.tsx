"use client";

import type { InsightCard as InsightCardType } from "@/lib/types";

const BADGE_STYLES: Record<string, { colors: string; icon: string; accent: string }> = {
  TREND: {
    colors: "bg-tertiary-fixed text-on-tertiary-fixed-variant",
    icon: "query_stats",
    accent: "from-tertiary/15 to-transparent",
  },
  OUTLIER: {
    colors: "bg-error-container text-on-error-container",
    icon: "warning",
    accent: "from-error/12 to-transparent",
  },
  CORRELATION: {
    colors: "bg-secondary-fixed text-on-secondary-fixed-variant",
    icon: "hub",
    accent: "from-secondary/12 to-transparent",
  },
  CAUSAL: {
    colors: "bg-primary-fixed text-on-primary-fixed-variant",
    icon: "account_tree",
    accent: "from-primary/12 to-transparent",
  },
  DISTRIBUTION: {
    colors: "bg-surface-container-high text-on-surface-variant",
    icon: "bar_chart",
    accent: "from-primary/8 to-transparent",
  },
  MISSING: {
    colors: "bg-surface-container text-on-surface-variant",
    icon: "data_alert",
    accent: "from-outline/10 to-transparent",
  },
};

const DEFAULT_BADGE = {
  colors: "bg-surface-container text-on-surface-variant",
  icon: "info",
  accent: "from-primary/8 to-transparent",
};

interface Props {
  insight: InsightCardType;
}

export default function InsightCard({ insight }: Props) {
  const badge = BADGE_STYLES[insight.type.toUpperCase()] || DEFAULT_BADGE;
  const confidence = insight.confidence == null ? null : Math.round(insight.confidence * 100);

  return (
    <div className="group rounded-[1.1rem] border border-outline-variant/30 bg-surface-container-low px-4 py-4 transition-colors hover:bg-surface-container">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-surface-container">
          <span className="material-symbols-outlined text-[19px] text-primary">{badge.icon}</span>
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2.5 py-1 text-[10px] font-bold tracking-[0.2em] uppercase ${badge.colors}`}>
              {insight.type}
            </span>
            {confidence != null && (
              <span className="rounded-full bg-surface-container px-2.5 py-1 text-[10px] font-semibold text-on-surface-variant">
                {confidence}% confidence
              </span>
            )}
          </div>

          <p className="mt-3 text-sm leading-6 text-on-surface">
            {insight.description}
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-on-surface-variant">
            <span>{insight.phase}</span>
            <span className="text-outline/60">•</span>
            <span>Rule #{insight.rule}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
