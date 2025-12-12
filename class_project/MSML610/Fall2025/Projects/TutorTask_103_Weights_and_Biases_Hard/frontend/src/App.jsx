import React, { useMemo, useState } from "react";

const DEFAULT_API = "http://localhost:8000";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API);
  const [ticker, setTicker] = useState("AAPL");
  const [lookbackDays, setLookbackDays] = useState(365);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, lookback_days: Number(lookbackDays) }),
      });
      if (!resp.ok) {
        throw new Error(`API error ${resp.status}: ${await resp.text()}`);
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
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
          <h3>Predict next close</h3>
          <label>
            Ticker
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} style={{ width: "100%", marginTop: 6 }} />
          </label>
          <label style={{ display: "block", marginTop: 12 }}>
            Lookback days
            <input
              type="number"
              value={lookbackDays}
              onChange={(e) => setLookbackDays(e.target.value)}
              style={{ width: "100%", marginTop: 6 }}
            />
          </label>

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


