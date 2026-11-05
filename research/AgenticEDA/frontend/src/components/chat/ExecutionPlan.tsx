"use client";

import { useNotebookStore } from "@/stores/notebookStore";
import type { PlanStepDetail } from "@/stores/notebookStore";

function DetailItem({ detail }: { detail: PlanStepDetail }) {
  return (
    <div className="flex items-start gap-1.5 ml-8 py-0.5">
      <div className="mt-[5px] shrink-0">
        {detail.status === "complete" ? (
          <svg className="w-2.5 h-2.5 text-primary/60" viewBox="0 0 10 10" fill="none">
            <path d="M2 5.5L4 7.5L8 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : detail.status === "current" ? (
          <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse ml-[2px]" />
        ) : (
          <div className="w-1.5 h-1.5 rounded-full bg-outline-variant/30 ml-[2px]" />
        )}
      </div>
      <p
        className={`text-[10px] leading-snug ${
          detail.status === "complete"
            ? "text-on-surface-variant/50 line-through"
            : detail.status === "current"
              ? "text-on-surface-variant font-medium"
              : "text-on-surface-variant/40"
        }`}
      >
        {detail.label}
      </p>
    </div>
  );
}

export default function ExecutionPlan() {
  const planSteps = useNotebookStore((s) => s.planSteps);

  const visibleSteps = planSteps.filter((s) => s.status !== "skipped");
  if (visibleSteps.length === 0) return null;

  return (
    <div className="p-4 bg-surface-container-lowest border border-outline-variant/10 rounded-2xl shadow-sm space-y-4 mb-4">
      {/* Header */}
      <div className="flex items-center gap-2 text-primary">
        <span className="material-symbols-outlined text-[18px]">list_alt</span>
        <span className="font-headline font-bold text-xs uppercase tracking-widest">
          Execution Plan
        </span>
      </div>

      {/* Stepper */}
      <div className="space-y-0">
        {visibleSteps.map((step, i) => {
          const isLast = i === visibleSteps.length - 1;
          const hasDetails = step.details && step.details.length > 0;
          // Only show details for current or recently-complete phases (not all past ones)
          const showDetails = hasDetails && (step.status === "current" || step.status === "upcoming");

          return (
            <div key={step.phase}>
              <div className="flex gap-3">
                <div className="flex flex-col items-center">
                  {/* Circle */}
                  {step.status === "complete" ? (
                    <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center">
                      <span className="material-symbols-outlined text-[14px] text-white">
                        check
                      </span>
                    </div>
                  ) : step.status === "current" ? (
                    <div className="w-5 h-5 rounded-full bg-primary/10 flex items-center justify-center ring-2 ring-primary">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-outline-variant/30 bg-transparent" />
                  )}
                  {/* Connector line — extend if details are shown */}
                  {!isLast && (
                    <div
                      className={`w-0.5 ${showDetails ? "h-2" : "h-6"} ${
                        step.status === "complete"
                          ? "bg-primary"
                          : "bg-outline-variant/20"
                      }`}
                    />
                  )}
                </div>
                {/* Label */}
                <div className="pt-0.5">
                  <p
                    className={`text-[11px] font-bold ${
                      step.status === "complete"
                        ? "text-on-surface-variant/60 line-through"
                        : step.status === "current"
                          ? "text-on-surface"
                          : "text-on-surface-variant opacity-50"
                    }`}
                  >
                    {step.phase}
                  </p>
                </div>
              </div>
              {/* Sub-items */}
              {showDetails && (
                <div className="pb-1">
                  {step.details!.map((d, j) => (
                    <DetailItem key={`${step.phase}-${j}`} detail={d} />
                  ))}
                  {/* Connector after details */}
                  {!isLast && (
                    <div className="flex justify-center ml-[7px] w-[2px]">
                      <div
                        className={`w-0.5 h-3 ${
                          step.status === "complete"
                            ? "bg-primary"
                            : "bg-outline-variant/20"
                        }`}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
