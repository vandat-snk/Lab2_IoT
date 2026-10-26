"""Telemetry failures must never be reported as successful database writes."""
import importlib
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend import database

api=importlib.import_module('backend.main')


def measurement():
    return {'request_id':str(uuid4()),'model_sha256':'a'*64,'inference_ms':98.4,
            'round_trip_ms':273.2,'end_to_end_ms':303.1,
            'counts':{'with_mask':2,'without_mask':1,'mask_weared_incorrect':1,'uncertain':0}}


@pytest.mark.parametrize('configured,result,status',[(False,None,503),(True,None,503),(True,42,200)])
def test_measurement_acknowledges_only_committed_writes(monkeypatch,configured,result,status):
    if configured:monkeypatch.setenv('DATABASE_URL','postgresql://configured')
    else:monkeypatch.delenv('DATABASE_URL',raising=False)
    monkeypatch.setattr(api,'save_measurement',lambda record:result)
    # This route does not need the CNN lifespan.
    response=TestClient(api.app).post('/api/measurements',json=measurement())
    assert response.status_code==status
    if status==200:
        assert response.json()=={'accepted':True,'saved':True,'record_id':42}
    else:
        assert 'saved' not in response.json()


def test_invalid_counts_never_reach_database(monkeypatch):
    monkeypatch.setenv('DATABASE_URL','postgresql://configured')
    def unexpected_write(record):
        pytest.fail('Invalid counts reached the database')
    monkeypatch.setattr(api,'save_measurement',unexpected_write)
    record=measurement();record['counts']['with_mask']=-1
    assert TestClient(api.app).post('/api/measurements',json=record).status_code==422


def test_real_postgresql_commit(monkeypatch):
    # Run only against an explicitly provided, disposable test database.
    dsn=os.getenv('MASKLAB_TEST_DATABASE_URL')
    if not dsn:pytest.skip('Set MASKLAB_TEST_DATABASE_URL for a disposable PostgreSQL database')
    import psycopg
    monkeypatch.setenv('DATABASE_URL',dsn)
    assert database.database_available() is True
    record=measurement()
    response=TestClient(api.app).post('/api/measurements',json=record)
    assert response.status_code==200
    record_id=response.json()['record_id']
    # A separate connection proves the insert committed and survives reconnect.
    with psycopg.connect(dsn) as conn:
        row=conn.execute('SELECT request_id, inference_ms, round_trip_ms, end_to_end_ms, counts '
                         'FROM lab_measurements WHERE id=%s',(record_id,)).fetchone()
        assert str(row[0])==record['request_id']
        assert row[1:4]==(98.4,273.2,303.1)
        assert row[4]==record['counts']
        columns=conn.execute("SELECT column_name FROM information_schema.columns "
                             "WHERE table_schema='public' AND table_name='lab_measurements'").fetchall()
        assert {c[0] for c in columns}=={'id','created_at','request_id','model_sha256',
                                       'inference_ms','round_trip_ms','end_to_end_ms','counts'}
