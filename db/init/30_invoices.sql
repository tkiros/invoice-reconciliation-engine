-- Invoice reconciliation tables. Runs once, on a fresh volume, after 20_app_schema.sql.

-- Reference data, as it would come from the ERP. Loaded by `make seed`.
CREATE TABLE IF NOT EXISTS products (
    sku          text PRIMARY KEY,
    description  text NOT NULL
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number  text PRIMARY KEY,
    vendor     text NOT NULL,
    po_date    date NOT NULL
);

CREATE TABLE IF NOT EXISTS po_lines (
    po_number   text    NOT NULL REFERENCES purchase_orders ON DELETE CASCADE,
    sku         text    NOT NULL REFERENCES products,
    qty         numeric NOT NULL,
    unit_price  numeric NOT NULL,
    PRIMARY KEY (po_number, sku)
);

-- One row per distinct uploaded file. The sha256 of the bytes is the idempotency key.
CREATE TABLE IF NOT EXISTS documents (
    id              bigserial   PRIMARY KEY,
    sha256          text        NOT NULL UNIQUE,
    filename        text        NOT NULL,
    pdf             bytea       NOT NULL,
    pages           int,                         -- NULL when the PDF can't be opened
    status          text        NOT NULL CHECK (status IN
                        ('received', 'extracted', 'auto_approved', 'needs_review', 'approved')),
    extractor       text,                        -- model name, or 'oracle' in the offline demo
    invoice         jsonb,                       -- the extracted invoice; corrections update it
    review_reasons  text[]      NOT NULL DEFAULT '{}',
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- What reconciliation and the arithmetic check found. Rewritten after each correction.
CREATE TABLE IF NOT EXISTS discrepancies (
    id           bigserial PRIMARY KEY,
    document_id  bigint    NOT NULL REFERENCES documents ON DELETE CASCADE,
    kind         text      NOT NULL,   -- price_mismatch, quantity_mismatch, missing_line, ...
    sku          text,                 -- NULL for document-level kinds
    detail       text      NOT NULL
);
CREATE INDEX IF NOT EXISTS discrepancies_document ON discrepancies (document_id);

-- Every state change and every field correction. The app only ever INSERTs here.
-- (CASCADE exists for dev resets such as `make demo`; the app never deletes documents.)
CREATE TABLE IF NOT EXISTS audit_log (
    id           bigserial   PRIMARY KEY,
    document_id  bigint      NOT NULL REFERENCES documents ON DELETE CASCADE,
    actor        text        NOT NULL,   -- 'system' or the reviewer's name
    action       text        NOT NULL,   -- received, extracted, needs_review, corrected, ...
    field        text,
    old_value    text,
    new_value    text,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_log_document ON audit_log (document_id);
