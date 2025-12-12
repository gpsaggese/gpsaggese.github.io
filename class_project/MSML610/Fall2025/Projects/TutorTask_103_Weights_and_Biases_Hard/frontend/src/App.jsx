import React, { useMemo, useState } from "react";

const DEFAULT_API = "http://localhost:8000";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API);
  const [mode, setMode] = useState("ticker"); // "ticker" | "manual"
  const [ticker, setTicker] = useState("AAPL");
  const [lookbackDays, setLookbackDays] = useState(365);
  const [horizonDays, setHorizonDays] = useState(1);
  const [manualJson, setManualJson] = useState('{\n  "Close": 200.0\n}');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let url = `${apiBase}/predict`;
      let body = { ticker, lookback_days: Number(lookbackDays), horizon_days: Number(horizonDays) };
      if (mode === "manual") {
        url = `${apiBase}/predict_features`;
        let parsed;
        try {
          parsed = JSON.parse(manualJson);
        } catch (e) {
          throw new Error("Manual features must be valid JSON (object/dict).");
        }
        body = { features: parsed, horizon_days: 1 };
      }

      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`API error ${resp.status}: ${txt}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial", padding: 24 }}>
      <h2>TutorTask103 – Stock Forecasting</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
          <h3>API Settings</h3>
          <label>
            API Base URL
            <input
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              style={{ width: "100%", marginTop: 6 }}
            />
          </label>
          <p style={{ marginTop: 12 }}>
            Health check: <a href={`${apiBase}/health`} target="_blank" rel="noreferrer">{`${apiBase}/health`}</a>
          </p>
          <p>
            Metrics: <a href={`${apiBase}/metrics`} target="_blank" rel="noreferrer">{`${apiBase}/metrics`}</a>
          </p>
          <p>
            Feature names:{" "}
            <a href={`${apiBase}/feature_names`} target="_blank" rel="noreferrer">{`${apiBase}/feature_names`}</a>
          </p>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
          <h3>Predict</h3>

          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <button
              onClick={() => setMode("ticker")}
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ccc", background: mode === "ticker" ? "#eef" : "white" }}
              disabled={loading}
            >
              Ticker mode
            </button>
            <button
              onClick={() => setMode("manual")}
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ccc", background: mode === "manual" ? "#eef" : "white" }}
              disabled={loading}
            >
              Manual feature mode
            </button>
          </div>

          {mode === "ticker" && (
            <>
              <label>
                Ticker
                <input value={ticker} onChange={(e) => setTicker(e.target.value)} style={{ width: "100%", marginTop: 6 }} />
              </label>
              <label style={{ display: "block", marginTop: 12 }}>
                Lookback days (must be ≥ 120)
                <input
                  type="number"
                  min={120}
                  value={lookbackDays}
                  onChange={(e) => setLookbackDays(e.target.value)}
                  style={{ width: "100%", marginTop: 6 }}
                />
              </label>
              <label style={{ display: "block", marginTop: 12 }}>
                Horizon days (1–30)
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={horizonDays}
                  onChange={(e) => setHorizonDays(e.target.value)}
                  style={{ width: "100%", marginTop: 6 }}
                />
              </label>
              <p style={{ marginTop: 10, color: "#555", fontSize: 13 }}>
                Note: multi-day forecast is iterative; only Close/return-based features are updated each step.
              </p>
            </>
          )}

          {mode === "manual" && (
            <>
              <p style={{ marginTop: 0, color: "#555", fontSize: 13 }}>
                Paste a JSON object with keys matching the model’s <code>feature_names</code> (see link on the left).
              </p>
              <textarea
                value={manualJson}
                onChange={(e) => setManualJson(e.target.value)}
                rows={10}
                style={{ width: "100%", marginTop: 6, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}
              />
              <p style={{ marginTop: 10, color: "#555", fontSize: 13 }}>
                Manual mode currently supports <b>1-step</b> prediction only.
              </p>
            </>
          )}

          <button onClick={predict} disabled={loading} style={{ marginTop: 12 }}>
            {loading ? "Running..." : "Predict"}
          </button>

          {error && <pre style={{ color: "crimson", marginTop: 12 }}>{error}</pre>}
          {result && (
            <pre style={{ marginTop: 12, background: "#f6f8fa", padding: 12, borderRadius: 8 }}>
{JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}


