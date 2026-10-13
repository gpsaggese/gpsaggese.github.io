type AboutPaneProps = {
  compact?: boolean;
};

const capabilities = [
  {
    icon: "upload_file",
    title: "Upload and intent",
    body: "Drop a CSV and tell the agent what you care about. The system reads the schema, infers column types, and starts planning an analysis route around your question.",
  },
  {
    icon: "account_tree",
    title: "Parallel investigation",
    body: "Focused subagents pursue hypotheses in parallel tracks, each writing executable notebook cells. They return concise evidence to the main workspace when done.",
  },
  {
    icon: "forum",
    title: "Mid-run guidance",
    body: "Send a message while the agent is working. Your direction gets queued and picked up at the next stable checkpoint, so the analysis shifts without losing progress.",
  },
  {
    icon: "auto_stories",
    title: "Evidence-linked reports",
    body: "The final story connects every claim back to the notebook cells, outputs, and plots that produced it. Nothing is asserted without a traceable source.",
  },
];

const steps = [
  { num: "01", label: "Schema read", accent: false },
  { num: "02", label: "Plan route", accent: false },
  { num: "03", label: "Generate cells", accent: false },
  { num: "04", label: "Human steering", accent: true },
  { num: "05", label: "Write report", accent: false },
];

function StepTimeline() {
  return (
    <div className="flex items-center gap-0 w-full">
      {steps.map((step, i) => (
        <div key={step.num} className="flex items-center flex-1 min-w-0">
          <div className="flex flex-col items-center gap-2">
            <div
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-xs font-bold font-mono transition-all ${
                step.accent
                  ? "bg-primary text-on-primary shadow-lg shadow-primary/25"
                  : "bg-surface-container-high text-on-surface"
              }`}
            >
              {step.num}
            </div>
            <span
              className={`text-[11px] font-semibold whitespace-nowrap ${
                step.accent ? "text-primary" : "text-on-surface-variant"
              }`}
            >
              {step.label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div className="flex-1 mx-2 h-px bg-gradient-to-r from-outline-variant/80 to-outline-variant/30" />
          )}
        </div>
      ))}
    </div>
  );
}

function TerminalDemo() {
  const lines = [
    { prefix: "$", text: "upload wind_farm_2024.csv", dim: false },
    { prefix: ">", text: 'intent: "Find underperforming turbines"', dim: false },
    { prefix: "", text: "Planning analysis route...", dim: true },
    { prefix: "", text: "Spawning subagent: power-curve-analysis", dim: true },
    { prefix: "", text: "Spawning subagent: downtime-correlation", dim: true },
    { prefix: "\u2709", text: 'Steering: "Focus on curtailment events"', dim: false },
    { prefix: "\u2713", text: "Report ready. 14 cells, 6 plots, 3 findings.", dim: false },
  ];

  return (
    <div className="rounded-xl border border-[#1e3a5f] bg-[#0a1929] font-mono text-[12.5px] leading-relaxed shadow-2xl overflow-hidden">
      <div className="flex items-center gap-2 border-b border-white/[0.06] px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
        <span className="ml-3 text-[10px] font-medium text-blue-200/40 tracking-wider uppercase">
          AgenticEDA session
        </span>
      </div>
      <div className="p-4 space-y-1.5">
        {lines.map((line, i) => (
          <div key={i} className={`flex gap-2 ${line.dim ? "text-blue-300/50" : "text-blue-100/90"}`}>
            {line.prefix && (
              <span
                className={`w-4 shrink-0 text-right ${
                  line.prefix === "\u2713"
                    ? "text-emerald-400"
                    : line.prefix === "\u2709"
                      ? "text-amber-300"
                      : "text-blue-400/70"
                }`}
              >
                {line.prefix}
              </span>
            )}
            {!line.prefix && <span className="w-4 shrink-0" />}
            <span>{line.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AboutPane({ compact = false }: AboutPaneProps) {
  return (
    <section
      className={`about-scroll-pane h-full overflow-y-auto ${compact ? "p-5 md:p-7" : "p-6 md:p-10"}`}
      style={{
        background:
          "radial-gradient(ellipse at 0% 0%, rgba(37,99,235,0.07), transparent 50%), radial-gradient(ellipse at 100% 100%, rgba(73,92,149,0.05), transparent 50%), linear-gradient(180deg, #f8f9ff 0%, #f0f4ff 100%)",
      }}
    >
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-12">
        {/* Hero */}
        <header className="flex flex-col gap-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary shadow-md shadow-primary/20">
              <span className="material-symbols-outlined text-[20px] text-on-primary">insights</span>
            </div>
            <span className="text-xs font-extrabold uppercase tracking-[0.25em] text-primary font-headline">
              AgenticEDA
            </span>
          </div>

          <h1 className="text-3xl font-extrabold leading-[1.1] tracking-tight text-on-surface font-headline md:text-[2.6rem]">
            Exploratory data analysis,
            <br />
            <span className="text-primary">guided by you.</span>
          </h1>

          <p className="max-w-xl text-[15px] leading-7 text-on-surface-variant">
            AgenticEDA turns a dataset and a question into a full analysis session. An AI agent plans the
            investigation, writes executable notebook cells, launches parallel research tracks, and compiles
            findings into a report you can trace back to the code that produced it.
          </p>

          <p className="max-w-xl text-[15px] leading-7 text-on-surface-variant">
            You stay in control throughout. Send guidance at any point during the run, and the agent
            adjusts course without losing context.
          </p>
        </header>

        {/* Terminal demo */}
        <TerminalDemo />

        {/* How it works */}
        <div className="flex flex-col gap-5">
          <h2 className="text-lg font-extrabold tracking-tight text-on-surface font-headline">
            How a session works
          </h2>
          <div className="rounded-xl border border-outline-variant/50 bg-white/70 backdrop-blur-sm p-6 shadow-sm">
            <StepTimeline />
          </div>
          <p className="text-sm leading-6 text-on-surface-variant">
            Step 04 is where you come in. While the agent is executing, you can redirect the analysis
            toward what matters most. Your message is queued and applied at the next checkpoint, keeping
            the run stable.
          </p>
        </div>

        {/* Capabilities grid */}
        <div className="flex flex-col gap-5">
          <h2 className="text-lg font-extrabold tracking-tight text-on-surface font-headline">
            What makes it different
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {capabilities.map((cap) => (
              <article
                key={cap.title}
                className="group rounded-xl border border-outline-variant/40 bg-white/80 backdrop-blur-sm p-5 shadow-sm transition-all hover:border-primary/30 hover:shadow-md hover:shadow-primary/[0.04]"
              >
                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-primary/[0.08]">
                  <span className="material-symbols-outlined text-[20px] text-primary">{cap.icon}</span>
                </div>
                <h3 className="text-[15px] font-bold text-on-surface font-headline">{cap.title}</h3>
                <p className="mt-2 text-[13px] leading-[1.65] text-on-surface-variant">{cap.body}</p>
              </article>
            ))}
          </div>
        </div>

        {/* Impact */}
        <div className="flex flex-col gap-5">
          <h2 className="text-lg font-extrabold tracking-tight text-on-surface font-headline">
            Why it matters
          </h2>

          {/* Hero metric */}
          <div className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/[0.06] to-transparent p-6 text-center">
            <p className="text-4xl font-extrabold tracking-tight text-primary font-headline md:text-5xl">
              30 hypotheses in 25 minutes
            </p>
            <p className="mt-3 text-lg font-bold text-primary/70 font-headline">99% less time</p>
            <p className="mt-1 text-sm text-on-surface-variant">
              A workload that typically takes a data scientist 1 to 2 weeks of manual exploration.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-outline-variant/40 bg-white/80 p-5">
              <p className="text-2xl font-extrabold text-on-surface font-headline">~66%</p>
              <p className="mt-1 text-[13px] leading-[1.6] text-on-surface-variant">
                of a data scientist&apos;s time goes to data prep and exploratory analysis.{" "}
                <a
                  href="https://www.anaconda.com/state-of-data-science-2020"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2"
                >
                  Anaconda, 2020
                </a>
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/40 bg-white/80 p-5">
              <p className="text-2xl font-extrabold text-on-surface font-headline">$112k</p>
              <p className="mt-1 text-[13px] leading-[1.6] text-on-surface-variant">
                median data scientist salary in the US.{" "}
                <a
                  href="https://www.bls.gov/ooh/math/data-scientists.htm"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2"
                >
                  BLS, 2024
                </a>
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/40 bg-white/80 p-5">
              <p className="text-2xl font-extrabold text-on-surface font-headline">~$740k</p>
              <p className="mt-1 text-[13px] leading-[1.6] text-on-surface-variant">
                in recovered productivity per year for a 10-person data team, reclaiming exploratory time at median salary.
              </p>
            </div>
          </div>

          <p className="text-[13px] leading-6 text-on-surface-variant">
            When EDA takes minutes instead of days, teams stop rationing which datasets get explored.
            Junior analysts produce senior-level investigation. Every finding ships with an audit trail
            for free.
          </p>
        </div>

        {/* Closing */}
        <footer className="border-t border-outline-variant/40 pt-6 pb-4">
          <p className="text-sm leading-7 text-on-surface-variant">
            The goal: data work as a collaborative loop. You bring the questions and the judgment.
            The agent handles the code, the bookkeeping, and the busywork of turning raw exploration
            into structured, auditable findings.
          </p>
          <div className="mt-5 flex items-center gap-2 text-xs text-on-surface-variant/70">
            <span className="material-symbols-outlined text-[14px]">code</span>
            <span className="font-medium">Built with Next.js and FastAPI</span>
          </div>
        </footer>
      </div>
    </section>
  );
}
