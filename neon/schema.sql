CREATE TABLE IF NOT EXISTS inference_results (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  source_url TEXT,
  source_type TEXT,
  frontend TEXT,
  backend TEXT,
  database_name TEXT,
  confidence NUMERIC(5,2),
  evidence JSONB,
  skeleton JSONB,
  raw_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_inference_results_created_at
  ON inference_results(created_at DESC);
