import React, { useEffect, useMemo, useState } from "react";

// Use Vite proxy by default: frontend calls /api/*, Vite forwards to FastAPI on :8000
const DEFAULT_API_BASE = "/api";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [apiStatus, setApiStatus] = useState({ ok: false, checked: false, detail: "" });

  const [mode, setMode] = useState("ticker"); // "ticker" | "manual"
  const [ticker, setTicker] = useState("AAPL");
  const [lookbackDays, setLookbackDays] = useState(365);
  const [horizonDays, setHorizonDays] = useState(1);

  const [featureNames, setFeatureNames] = useState([]);
  const [featureFilter, setFeatureFilter] = useState("");
  const [featureValues, setFeatureValues] = useState({});
  const [manualJson, setManualJson] = useState("");
  const [manualEditorMode, setManualEditorMode] = useState("form"); // "form" | "json"
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const api = useMemo(() => {
    // normalize trailing slash
    const b = (apiBase || "").trim();
    return b.endsWith("/") ? b.slice(0, -1) : b;
  }, [apiBase]);

  const checkHealth = async () => {
    setApiStatus({ ok: false, checked: false, detail: "Checking..." });
    try {
      const resp = await fetch(`${api}/health`, { method: "GET" });
      if (!resp.ok) {
        const txt = await resp.text();
        setApiStatus({ ok: false, checked: true, detail: `HTTP ${resp.status}: ${txt}` });
        return;
      }
      const data = await resp.json().catch(() => ({}));
      setApiStatus({ ok: true, checked: true, detail: JSON.stringify(data) });
    } catch (e) {
      setApiStatus({ ok: false, checked: true, detail: String(e) });
    }
  };

  const loadManualFeatures = async () => {
    setError(null);
    try {
      const resp = await fetch(`${api}/feature_names`);
      if (!resp.ok) throw new Error(`feature_names failed: ${resp.status} ${await resp.text()}`);
      const data = await resp.json();
      const names = Array.isArray(data.feature_names) ? data.feature_names : [];
      setFeatureNames(names);

      const resp2 = await fetch(`${api}/feature_template`);
      if (!resp2.ok) throw new Error(`feature_template failed: ${resp2.status} ${await resp2.text()}`);
      const data2 = await resp2.json();
      const feats = data2.features || {};
      setFeatureValues(feats);
      setManualJson(JSON.stringify(feats, null, 2));
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    checkHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  useEffect(() => {
    if (mode === "manual" && featureNames.length === 0) {
      loadManualFeatures();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let url = `${api}/predict`;
      let body = { ticker, lookback_days: Number(lookbackDays), horizon_days: Number(horizonDays) };
      if (mode === "manual") {
        url = `${api}/predict_features`;
        let parsed;
        try {
          parsed = manualEditorMode === "form" ? featureValues : JSON.parse(manualJson);
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

  const filteredFeatureNames = useMemo(() => {
    const q = (featureFilter || "").trim().toLowerCase();
    if (!q) return featureNames;
    return featureNames.filter((n) => String(n).toLowerCase().includes(q));
  }, [featureNames, featureFilter]);

  return (
    <div className="container">
      <div className="header">
        <div>
          <div className="title">TutorTask103 – Stock Forecasting</div>
          <div className="subtitle">
            Ticker-based iterative forecast (N days) + manual feature prediction using the trained Linear Regression model.
          </div>
        </div>
        <div className="pill">
          API: <span className={apiStatus.ok ? "ok" : "bad"}>{apiStatus.ok ? "ONLINE" : "OFFLINE"}</span>
          <button className="btn" onClick={checkHealth} disabled={loading}>
            Recheck
          </button>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h3>API Settings</h3>
          <div className="field">
            <div className="label">API Base URL</div>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
          </div>
          <div className="note">
            Default is <span className="mono">/api</span> (Vite proxy). If your backend is reachable directly, you can set it to{" "}
            <span className="mono">http://localhost:8000</span>.
          </div>

          <div style={{ marginTop: 10, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <a className="btn" href={`${api}/health`} target="_blank" rel="noreferrer">
              /health
            </a>
            <a className="btn" href={`${api}/metrics`} target="_blank" rel="noreferrer">
              /metrics
            </a>
            <a className="btn" href={`${api}/feature_names`} target="_blank" rel="noreferrer">
              /feature_names
            </a>
            <a className="btn" href={`${api}/feature_template`} target="_blank" rel="noreferrer">
              /feature_template
            </a>
          </div>

          {apiStatus.checked && (
            <div className="note" style={{ marginTop: 10 }}>
              Status detail: <span className="mono">{apiStatus.detail}</span>
            </div>
          )}

          <div className="note" style={{ marginTop: 12 }}>
            If you still see <b>Failed to fetch</b>, your FastAPI server is not exposed to the host. Start the container with{" "}
            <span className="mono">-p 8000:8000</span>.
          </div>
        </div>

        <div className="card">
          <h3>Predict</h3>

          <div className="tabs">
            <button className={`tab ${mode === "ticker" ? "tabActive" : ""}`} onClick={() => setMode("ticker")} disabled={loading}>
              Ticker mode
            </button>
            <button className={`tab ${mode === "manual" ? "tabActive" : ""}`} onClick={() => setMode("manual")} disabled={loading}>
              Manual feature mode
            </button>
          </div>

          {mode === "ticker" && (
            <>
              <div className="row">
                <div className="field">
                  <div className="label">Ticker</div>
                  <input value={ticker} onChange={(e) => setTicker(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Horizon days (1–120)</div>
                  <input type="number" min={1} max={120} value={horizonDays} onChange={(e) => setHorizonDays(e.target.value)} />
                </div>
              </div>

              <div className="field">
                <div className="label">Lookback days (must be ≥ 120)</div>
                <input type="number" min={120} value={lookbackDays} onChange={(e) => setLookbackDays(e.target.value)} />
              </div>

              <div className="note">
                Multi-day forecast is iterative: we update only <span className="mono">Close</span>/<span className="mono">return</span>-based
                features each step; other indicators are held constant.
              </div>
            </>
          )}

          {mode === "manual" && (
            <>
              <div className="tabs">
                <button
                  className={`tab ${manualEditorMode === "form" ? "tabActive" : ""}`}
                  onClick={() => setManualEditorMode("form")}
                  disabled={loading}
                >
                  Form
                </button>
                <button
                  className={`tab ${manualEditorMode === "json" ? "tabActive" : ""}`}
                  onClick={() => setManualEditorMode("json")}
                  disabled={loading}
                >
                  JSON
                </button>
                <button className="btn" onClick={loadManualFeatures} disabled={loading}>
                  Load feature template
                </button>
              </div>

              <div className="note">
                Manual mode uses your trained LR model. It supports <b>1-step</b> prediction only. Use <span className="mono">/feature_template</span>{" "}
                to start.
              </div>

              {manualEditorMode === "json" && (
                <div className="field" style={{ marginTop: 10 }}>
                  <div className="label">Features JSON</div>
                  <textarea rows={10} value={manualJson} onChange={(e) => setManualJson(e.target.value)} />
                </div>
              )}

              {manualEditorMode === "form" && (
                <>
                  <div className="field" style={{ marginTop: 10 }}>
                    <div className="label">Search features</div>
                    <input value={featureFilter} onChange={(e) => setFeatureFilter(e.target.value)} placeholder="e.g., rsi, lag, ma_" />
                  </div>
                  <div className="featureList">
                    {filteredFeatureNames.length === 0 && <div className="note">No features loaded yet. Click “Load feature template”.</div>}
                    {filteredFeatureNames.slice(0, 60).map((name) => (
                      <div key={name} className="featureRow">
                        <div className="mono">{name}</div>
                        <input
                          type="number"
                          value={featureValues[name] ?? 0}
                          onChange={(e) => {
                            const v = e.target.value;
                            setFeatureValues((prev) => ({ ...prev, [name]: v === "" ? 0 : Number(v) }));
                          }}
                        />
                      </div>
                    ))}
                    {filteredFeatureNames.length > 60 && (
                      <div className="note" style={{ marginTop: 10 }}>
                        Showing first 60 matches. Use search to narrow down.
                      </div>
                    )}
                  </div>
                </>
              )}
            </>
          )}

          <button className="btn btnPrimary" onClick={predict} disabled={loading || !apiStatus.ok} style={{ marginTop: 12 }}>
            {loading ? "Running..." : "Predict"}
          </button>

          {error && <div className="error">{error}</div>}
          {result && <div className="result">{JSON.stringify(result, null, 2)}</div>}
        </div>
      </div>
    </div>
  );
}


