-- Schema for fraud detection audit + prediction history.
-- Loaded automatically by postgres container on first start
-- (mounted to /docker-entrypoint-initdb.d/init.sql).

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,          -- 'predict' | 'predict_by_id'
    transaction_id  BIGINT,                 -- NULL for manual /predict
    fraud_proba     DOUBLE PRECISION NOT NULL,
    predicted_label SMALLINT NOT NULL,
    model           TEXT NOT NULL,
    model_version   TEXT,
    request_payload JSONB
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions (ts DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_txn ON predictions (transaction_id)
    WHERE transaction_id IS NOT NULL;


CREATE TABLE IF NOT EXISTS decisions (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transaction_id  BIGINT,                 -- NULL for what-if
    source          TEXT NOT NULL,          -- 'by_id' | 'whatif'
    decision        TEXT NOT NULL,          -- approve_legit | confirm_fraud | need_more_info
    model_proba     DOUBLE PRECISION,
    model_label     SMALLINT,
    model           TEXT,
    note            TEXT,
    prediction_id   INTEGER REFERENCES predictions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions (ts DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions (decision);


-- Bulk prediction jobs (CSV upload from UI).
-- Each uploaded CSV row gets one row here. Status transitions:
--   pending -> completed | failed
-- After processing, the user can label rows via Bulk Audit page.

CREATE TABLE IF NOT EXISTS bulk_predictions (
    id              SERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL,                -- groups rows from same upload
    job_filename    TEXT,                           -- source CSV name
    row_idx         INTEGER,                        -- position in source CSV
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- input
    transaction_id  BIGINT,                         -- optional (NULL for manual rows)
    request_payload JSONB NOT NULL,                 -- raw features from CSV row

    -- processing state
    status          TEXT NOT NULL DEFAULT 'pending', -- pending | completed | failed
    fraud_proba     DOUBLE PRECISION,
    predicted_label SMALLINT,
    model           TEXT,
    error           TEXT,

    -- user labeling (Bulk Audit)
    user_label      TEXT,                           -- fraud | legit | review | blocked
    user_note       TEXT,
    labeled_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bulk_job ON bulk_predictions (job_id);
CREATE INDEX IF NOT EXISTS idx_bulk_status ON bulk_predictions (status);
CREATE INDEX IF NOT EXISTS idx_bulk_user_label ON bulk_predictions (user_label)
    WHERE user_label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_bulk_ts ON bulk_predictions (ts DESC);


-- Continuous training audit trail.
-- Each retrain job (from reviewer-corrected labels) gets a row here.
-- Lifecycle: trained -> promoted | rejected.

CREATE TABLE IF NOT EXISTS training_jobs (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Windowing strategy
    window_strategy        TEXT,                          -- 'sliding_30d+history_20pct'
    window_recent_days     INTEGER DEFAULT 30,
    window_history_sample  DOUBLE PRECISION DEFAULT 0.20,
    train_cutoff_date      TIMESTAMPTZ,                   -- boundary "old" vs "new"

    -- Data composition
    n_new_labels    INTEGER NOT NULL,                     -- recent labeled rows used
    n_old_labels    INTEGER NOT NULL,                     -- sampled history rows
    fraud_rate      DOUBLE PRECISION,                     -- positive rate in training set

    -- Model output
    model_version   TEXT NOT NULL,                        -- 'v2', 'v3', ...
    model_path      TEXT NOT NULL,                        -- 'models/lgbm_v2.pkl'
    val_auc         DOUBLE PRECISION,
    train_seconds   DOUBLE PRECISION,

    -- Promotion lifecycle
    status          TEXT NOT NULL DEFAULT 'trained',      -- trained | promoted | rejected
    promoted_at     TIMESTAMPTZ,
    promoted_by     TEXT,                                 -- user/system trigger
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_ts ON training_jobs (ts DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs (status);
