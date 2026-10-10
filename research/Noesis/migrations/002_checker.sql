-- M2 checker persistence. Preserve M0/M1 tables while adding versioned
-- canaries, probe scheduling, and explainable verdict evidence.

ALTER TABLE contracts
    ADD COLUMN q_min DOUBLE PRECISION NOT NULL DEFAULT 0.85
        CHECK (q_min BETWEEN 0 AND 1),
    ADD COLUMN canaries_target INT NOT NULL DEFAULT 10
        CHECK (canaries_target > 0),
    ADD COLUMN window_max_seconds INT NOT NULL DEFAULT 300
        CHECK (window_max_seconds > 0),
    ADD COLUMN checker_policy TEXT NOT NULL DEFAULT 'm2-v1';

ALTER TABLE canaries
    ADD COLUMN bank_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN family TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN generator_seed BIGINT,
    ADD COLUMN status TEXT NOT NULL DEFAULT 'DRAFT'
        CHECK (status IN ('DRAFT', 'CALIBRATED', 'ACTIVE', 'RETIRED')),
    ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX canaries_bank_status_idx
    ON canaries (bank_version, status, family);

CREATE TABLE canary_calibration_runs (
    run_id          BIGSERIAL PRIMARY KEY,
    bank_version    TEXT NOT NULL,
    healthy_model   TEXT NOT NULL,
    weak_model      TEXT NOT NULL,
    repetitions     INT NOT NULL CHECK (repetitions > 0),
    state           TEXT NOT NULL DEFAULT 'RUNNING'
                    CHECK (state IN ('RUNNING', 'COMPLETED', 'FAILED')),
    configuration   JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary         JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE canary_calibration_results (
    result_id       BIGSERIAL PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES canary_calibration_runs (run_id)
                    ON DELETE CASCADE,
    canary_id       TEXT NOT NULL REFERENCES canaries (canary_id),
    target_role     TEXT NOT NULL CHECK (target_role IN ('HEALTHY', 'WEAK')),
    provider_slug   TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    attempt         INT NOT NULL CHECK (attempt > 0),
    provider_status TEXT NOT NULL,
    response        TEXT,
    correct         BOOLEAN,
    latency_ms      DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    cost_usd        DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, canary_id, target_role, provider_slug, model_id, attempt)
);
CREATE INDEX canary_calibration_results_run_idx
    ON canary_calibration_results (run_id, canary_id);

ALTER TABLE requests
    ADD COLUMN started_at TIMESTAMPTZ,
    ADD COLUMN completed_at TIMESTAMPTZ,
    ADD COLUMN grader_version TEXT;

UPDATE requests
SET completed_at = created_at,
    started_at = created_at - latency_ms * interval '1 millisecond'
WHERE started_at IS NULL OR completed_at IS NULL;

ALTER TABLE requests
    ALTER COLUMN started_at SET NOT NULL,
    ALTER COLUMN started_at SET DEFAULT now(),
    ALTER COLUMN completed_at SET NOT NULL,
    ALTER COLUMN completed_at SET DEFAULT now();

ALTER TABLE requests
    ADD CONSTRAINT requests_canary_fields_check CHECK (
        (is_canary AND canary_id IS NOT NULL)
        OR
        (NOT is_canary AND canary_id IS NULL AND canary_correct IS NULL
                       AND grader_version IS NULL)
    );

CREATE INDEX requests_contract_canary_started_idx
    ON requests (contract_id, is_canary, started_at);

CREATE TABLE canary_probes (
    probe_id        BIGSERIAL PRIMARY KEY,
    contract_id     BIGINT NOT NULL REFERENCES contracts (contract_id)
                    ON DELETE CASCADE,
    probe_slot      INT NOT NULL CHECK (probe_slot > 0),
    canary_id       TEXT NOT NULL REFERENCES canaries (canary_id),
    state           TEXT NOT NULL DEFAULT 'SCHEDULED'
                    CHECK (state IN ('SCHEDULED', 'RUNNING', 'COMPLETED',
                                     'FAILED_INTERNAL')),
    request_id      BIGINT UNIQUE REFERENCES requests (request_id),
    grader_version  TEXT,
    error           TEXT,
    scheduled_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    UNIQUE (contract_id, probe_slot)
);
CREATE INDEX canary_probes_contract_state_idx
    ON canary_probes (contract_id, state);

ALTER TABLE contract_verdicts
    ADD COLUMN latency_successes INT NOT NULL DEFAULT 0
        CHECK (latency_successes >= 0),
    ADD COLUMN quality_successes INT NOT NULL DEFAULT 0
        CHECK (quality_successes >= 0),
    ADD COLUMN checker_version TEXT NOT NULL DEFAULT 'm2-v1',
    ADD COLUMN bank_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN checker_disposition TEXT NOT NULL DEFAULT 'VALID'
        CHECK (checker_disposition IN ('VALID', 'CHECKER_UNHEALTHY')),
    ADD COLUMN reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN excluded_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN window_start TIMESTAMPTZ,
    ADD COLUMN window_end TIMESTAMPTZ;

ALTER TABLE contract_verdicts
    ADD CONSTRAINT contract_verdict_latency_count_check
        CHECK (latency_successes <= n),
    ADD CONSTRAINT contract_verdict_quality_count_check
        CHECK (quality_successes <= n_canary);

CREATE TABLE verdict_evidence (
    contract_id       BIGINT NOT NULL REFERENCES contract_verdicts (contract_id)
                      ON DELETE CASCADE,
    request_id        BIGINT NOT NULL REFERENCES requests (request_id),
    latency_eligible  BOOLEAN NOT NULL,
    latency_success   BOOLEAN,
    latency_reason    TEXT NOT NULL,
    quality_eligible  BOOLEAN NOT NULL,
    quality_success   BOOLEAN,
    quality_reason    TEXT NOT NULL,
    PRIMARY KEY (contract_id, request_id),
    CHECK (
        (latency_eligible AND latency_success IS NOT NULL)
        OR (NOT latency_eligible AND latency_success IS NULL)
    ),
    CHECK (
        (quality_eligible AND quality_success IS NOT NULL)
        OR (NOT quality_eligible AND quality_success IS NULL)
    )
);
