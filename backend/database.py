"""Optional PostgreSQL telemetry. Never stores camera images or face crops."""
import os
import json
import logging
from datetime import datetime, timezone

LOG=logging.getLogger(__name__)


def save_measurement(record):
    dsn=os.getenv('DATABASE_URL')
    if not dsn:return False
    try:
        import psycopg
        with psycopg.connect(dsn,connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS lab_measurements (
                    id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL,
                    request_id UUID NOT NULL, model_sha256 TEXT NOT NULL,
                    inference_ms DOUBLE PRECISION NOT NULL, round_trip_ms DOUBLE PRECISION NOT NULL,
                    end_to_end_ms DOUBLE PRECISION NOT NULL, counts JSONB NOT NULL)''')
                cur.execute('''INSERT INTO lab_measurements
                    (created_at,request_id,model_sha256,inference_ms,round_trip_ms,end_to_end_ms,counts)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (datetime.now(timezone.utc),record['request_id'],record['model_sha256'],record['inference_ms'],
                     record['round_trip_ms'],record['end_to_end_ms'],json.dumps(record['counts'])))
        return True
    except Exception:
        LOG.warning('PostgreSQL telemetry unavailable')
        return False
