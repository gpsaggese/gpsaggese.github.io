-- Standalone monitoring persistence. These records are deliberately separate
-- from contract evidence, verdicts, and seller reputation.

CREATE TABLE monitoring_runs (
    run_id          BIGSERIAL PRIMARY KEY,
    bank_version    TEXT NOT NULL,
    provider_slug   TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'RUNNING'
                    CHECK (state IN ('RUNNING', 'COMPLETED', 'FAILED')),
    configuration   JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary         JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX monitoring_runs_series_started_idx
    ON monitoring_runs (provider_slug, model_id, bank_version, started_at);

CREATE TABLE monitoring_observations (
    observation_id   BIGSERIAL PRIMARY KEY,
    run_id           BIGINT NOT NULL REFERENCES monitoring_runs (run_id)
                     ON DELETE CASCADE,
    canary_id        TEXT NOT NULL REFERENCES canaries (canary_id),
    provider_status  TEXT NOT NULL,
    response         TEXT,
    correct          BOOLEAN,
    grader_reason    TEXT NOT NULL,
    grader_version   TEXT NOT NULL,
    requested_provider TEXT NOT NULL,
    served_by        TEXT,
    requested_model TEXT NOT NULL,
    model_served     TEXT,
    latency_ms       DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    prompt_tokens    INT CHECK (prompt_tokens >= 0),
    completion_tokens INT CHECK (completion_tokens >= 0),
    cost_usd         DOUBLE PRECISION NOT NULL DEFAULT 0
                     CHECK (cost_usd >= 0),
    http_status      INT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, canary_id)
);
CREATE INDEX monitoring_observations_run_idx
    ON monitoring_observations (run_id, canary_id);
