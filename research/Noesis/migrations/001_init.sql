-- Noesis MVP schema (PRD §7). Shared contract between gateway/metering and
-- market/reputation. Idempotent: safe to run on every startup.

CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK (kind IN ('buyer', 'seller', 'operator')),
    api_key_hash    TEXT NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sellers (
    seller_id            TEXT PRIMARY KEY,
    account_id           TEXT NOT NULL REFERENCES accounts (account_id),
    provider_slug        TEXT NOT NULL,          -- OpenRouter endpoint tag, e.g. 'groq'
    provider_name        TEXT NOT NULL,          -- name OpenRouter reports back, e.g. 'Groq'
    model_id             TEXT NOT NULL,
    state                TEXT NOT NULL DEFAULT 'eligible'
                         CHECK (state IN ('eligible', 'blocked', 'probation')),
    reputation           DOUBLE PRECISION NOT NULL DEFAULT 0.8
                         CHECK (reputation BETWEEN 0 AND 1),
    blocked_until_round  BIGINT,
    probation_passes     INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rounds (
    round_id              BIGSERIAL PRIMARY KEY,
    tier                  TEXT NOT NULL,
    started_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    cleared_at            TIMESTAMPTZ,
    clearing_price        DOUBLE PRECISION,
    matched_n_tasks       INT NOT NULL DEFAULT 0,
    unfilled_bid_n_tasks  INT NOT NULL DEFAULT 0,
    unfilled_ask_n_tasks  INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bids (
    order_id      BIGSERIAL PRIMARY KEY,
    account_id    TEXT NOT NULL REFERENCES accounts (account_id),
    n_tasks       INT NOT NULL CHECK (n_tasks > 0),
    c_level_min   TEXT NOT NULL,
    l_max         DOUBLE PRECISION NOT NULL CHECK (l_max > 0),
    r_min         DOUBLE PRECISION NOT NULL CHECK (r_min BETWEEN 0 AND 1),
    p_max         DOUBLE PRECISION NOT NULL CHECK (p_max >= 0),
    submitted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    round_id      BIGINT REFERENCES rounds (round_id)   -- set when processed
);

CREATE TABLE IF NOT EXISTS asks (
    order_id      BIGSERIAL PRIMARY KEY,
    seller_id     TEXT NOT NULL REFERENCES sellers (seller_id),
    n_tasks       INT NOT NULL CHECK (n_tasks > 0),
    c_level       TEXT NOT NULL,
    l_typical     DOUBLE PRECISION NOT NULL CHECK (l_typical > 0),
    r_typical     DOUBLE PRECISION NOT NULL CHECK (r_typical BETWEEN 0 AND 1),
    p_min         DOUBLE PRECISION NOT NULL CHECK (p_min >= 0),
    submitted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    standing      BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id       BIGSERIAL PRIMARY KEY,
    round_id          BIGINT NOT NULL REFERENCES rounds (round_id),
    bid_order_id      BIGINT NOT NULL REFERENCES bids (order_id),
    ask_order_id      BIGINT NOT NULL REFERENCES asks (order_id),
    buyer_account_id  TEXT NOT NULL REFERENCES accounts (account_id),
    seller_id         TEXT NOT NULL REFERENCES sellers (seller_id),
    n_tasks           INT NOT NULL CHECK (n_tasks > 0),
    c_level           TEXT NOT NULL,
    l_max             DOUBLE PRECISION NOT NULL,
    r_min             DOUBLE PRECISION NOT NULL,
    price             DOUBLE PRECISION NOT NULL,
    state             TEXT NOT NULL DEFAULT 'PENDING'
                      CHECK (state IN ('PENDING', 'ACTIVE', 'CLOSED',
                                       'PASSED', 'FAILED', 'INSUFFICIENT')),
    window_start      TIMESTAMPTZ,
    window_end        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS contracts_state_idx ON contracts (state);
CREATE INDEX IF NOT EXISTS contracts_seller_idx ON contracts (seller_id);

CREATE TABLE IF NOT EXISTS canaries (
    canary_id    TEXT PRIMARY KEY,
    prompt       TEXT NOT NULL,
    answer       TEXT NOT NULL,
    normalizer   TEXT NOT NULL DEFAULT 'alnum_lower',
    calibrated   BOOLEAN NOT NULL DEFAULT FALSE,
    healthy_acc  DOUBLE PRECISION,
    weak_acc     DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS requests (
    request_id            BIGSERIAL PRIMARY KEY,
    contract_id           BIGINT REFERENCES contracts (contract_id),
    seller_id             TEXT REFERENCES sellers (seller_id),
    pinned_provider       TEXT NOT NULL,
    actual_provider       TEXT,
    attribution_mismatch  BOOLEAN NOT NULL DEFAULT FALSE,
    model_requested       TEXT NOT NULL,
    model_served          TEXT,
    is_canary             BOOLEAN NOT NULL DEFAULT FALSE,
    canary_id             TEXT REFERENCES canaries (canary_id),
    canary_correct        BOOLEAN,
    prompt                TEXT NOT NULL,
    completion            TEXT,
    prompt_tokens         INT,
    completion_tokens     INT,
    latency_ms            DOUBLE PRECISION NOT NULL,
    status                TEXT NOT NULL
                          -- Only 'timeout' and 'upstream_error' count against the seller.
                          CHECK (status IN ('ok', 'timeout', 'upstream_error',
                                            'buyer_error', 'gateway_error',
                                            'refused_cap')),
    cost_usd              DOUBLE PRECISION NOT NULL DEFAULT 0,
    fault_injected        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS requests_contract_idx ON requests (contract_id);
CREATE INDEX IF NOT EXISTS requests_created_idx ON requests (created_at);

CREATE TABLE IF NOT EXISTS contract_verdicts (
    contract_id   BIGINT PRIMARY KEY REFERENCES contracts (contract_id),
    n             INT NOT NULL,
    r_lat         DOUBLE PRECISION,
    r_lat_upper   DOUBLE PRECISION,
    n_canary      INT NOT NULL,
    q             DOUBLE PRECISION,
    q_upper       DOUBLE PRECISION,
    verdict       TEXT NOT NULL CHECK (verdict IN ('PASSED', 'FAILED', 'INSUFFICIENT')),
    reason        TEXT NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reputation_events (
    id            BIGSERIAL PRIMARY KEY,
    seller_id     TEXT NOT NULL REFERENCES sellers (seller_id),
    contract_id   BIGINT REFERENCES contracts (contract_id),
    round_id      BIGINT REFERENCES rounds (round_id),
    rho_before    DOUBLE PRECISION NOT NULL,
    rho_after     DOUBLE PRECISION NOT NULL,
    state_before  TEXT NOT NULL,
    state_after   TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
