"""Optional PostgreSQL telemetry. Never stores camera images or face crops."""
import os
import json
import logging
import time
from threading import Lock
from datetime import datetime, timezone

LOG=logging.getLogger(__name__)
_status_lock=Lock()
_status={'dsn':None,'until':0,'available':False}


def database_available():
    """Check the actual connection; keep health probes cheap for 15 seconds."""
    dsn=os.getenv('DATABASE_URL')
    if not dsn:return False
    with _status_lock:
        if _status['dsn']==dsn and time.monotonic()<_status['until']:
            return _status['available']
        available=False
        try:
            import psycopg
            with psycopg.connect(dsn,connect_timeout=3,options='-c statement_timeout=3000') as conn:
                conn.execute('SELECT 1')
            available=True
        except Exception:
            LOG.warning('PostgreSQL connection unavailable')
        _status.update(dsn=dsn,until=time.monotonic()+15,available=available)
        return available


def save_measurement(record):
    dsn=os.getenv('DATABASE_URL')
    if not dsn:return None
    try:
        import psycopg
        with psycopg.connect(dsn,connect_timeout=3,options='-c statement_timeout=3000') as conn:
            with conn.cursor() as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS lab_measurements (
                    id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL,
                    request_id UUID NOT NULL, model_sha256 TEXT NOT NULL,
                    inference_ms DOUBLE PRECISION NOT NULL, round_trip_ms DOUBLE PRECISION NOT NULL,
                    end_to_end_ms DOUBLE PRECISION NOT NULL, counts JSONB NOT NULL)''')
                cur.execute('''INSERT INTO lab_measurements
                    (created_at,request_id,model_sha256,inference_ms,round_trip_ms,end_to_end_ms,counts)
                    VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                    (datetime.now(timezone.utc),record['request_id'],record['model_sha256'],record['inference_ms'],
                     record['round_trip_ms'],record['end_to_end_ms'],json.dumps(record['counts'])))
                record_id=cur.fetchone()[0]
        # The connection context has committed before reporting success.
        return record_id
    except Exception as exc:
        # Log a diagnostic code without leaking the connection string or data.
        LOG.warning('PostgreSQL telemetry unavailable (%s, SQLSTATE %s)',type(exc).__name__,getattr(exc,'sqlstate',None))
        return None
