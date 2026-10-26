from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.inference import ModelEngine
from backend.geometry import padded_box

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:yield c


def test_crop_clamps():
    assert padded_box([0,0,100,100],100,100)==[0,0,100,100]


def test_split_has_no_source_or_group_leakage():
    sources=json.loads((ROOT/'artifacts/source_split.json').read_text())
    manifest=[{'source':s['source'],'group':s['group'],'split':s['split'],'label':f['label']} for s in sources for f in s['faces']]
    for key in ['source','group']:
        partitions={s:{r[key] for r in manifest if r['split']==s} for s in ['train','val','test']}
        assert partitions['train'].isdisjoint(partitions['val'])
        assert partitions['train'].isdisjoint(partitions['test'])
        assert partitions['val'].isdisjoint(partitions['test'])
    counts={label:sum(r['label']==label for r in manifest) for label in ['with_mask','without_mask','mask_weared_incorrect']}
    assert min(counts.values())>=20


def test_health_model_and_frontend(client):
    health=client.get('/api/health');assert health.status_code==200
    assert len(health.json()['labels'])==3
    assert len(health.json()['model_sha256'])==64
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200
    import hashlib
    bundle=client.get('/api/model/download')
    assert bundle.status_code==200
    assert hashlib.sha256(bundle.content).hexdigest()==health.json()['model_sha256']


def test_blank_frame_and_timers(client):
    buf=BytesIO();Image.new('RGB',(640,480),'white').save(buf,format='JPEG')
    response=client.post('/api/predict',files={'file':('blank.jpg',buf.getvalue(),'image/jpeg')})
    assert response.status_code==200
    result=response.json();assert result['face_count']==0;assert not result['detections']
    assert sum(result['counts'].values())==0
    assert result['timing']['server_processing_ms']>=result['timing']['inference_ms']>=0
    assert result['timing']['cnn_ms']==0


@pytest.mark.parametrize('payload',[b'',b'not an image',b'\x00'*100])
def test_bad_images(client,payload):
    assert client.post('/api/predict',files={'file':('bad.jpg',payload,'image/jpeg')}).status_code==400


def test_size_and_threshold_validation(client):
    assert client.post('/api/predict',files={'file':('big.jpg',b'x'*(8*1024*1024+1),'image/jpeg')}).status_code==413
    assert client.post('/api/predict?classification_threshold=2',files={'file':('a.jpg',b'x','image/jpeg')}).status_code==422
    buf=BytesIO();Image.new('RGB',(5000,2)).save(buf,format='PNG')
    assert client.post('/api/predict',files={'file':('wide.png',buf.getvalue(),'image/png')}).status_code==413


def test_bundle_corruption_rejected(tmp_path):
    source=ROOT/'models/mask_detector.bundle';dest=tmp_path/'corrupt.bundle'
    with zipfile.ZipFile(source) as zin,zipfile.ZipFile(dest,'w') as zout:
        for name in zin.namelist():
            data=zin.read(name)
            if name=='classifier.onnx':data+=b'corrupt'
            zout.writestr(name,data)
    with pytest.raises(ValueError,match='Corrupt model'):
        ModelEngine(dest)


def test_real_group_frame_recognizes_all_three_classes(client):
    import cv2
    from scripts.evaluate_pipeline import iou
    sample=cv2.imread(str(ROOT/'web/samples/three_classes.jpg'))
    # Match the browser's JPEG quality, rather than testing only a pristine PNG.
    ok,jpeg=cv2.imencode('.jpg',sample,[cv2.IMWRITE_JPEG_QUALITY,88])
    assert ok
    response=client.post('/api/predict',files={'file':('group.jpg',jpeg.tobytes(),'image/jpeg')})
    assert response.status_code==200
    predictions=response.json()['detections']
    demo=json.loads((ROOT/'artifacts/demo_sample.json').read_text())
    source=next(s for s in json.loads((ROOT/'artifacts/source_split.json').read_text()) if s['source']==demo['source'])
    found={f['label'] for f in source['faces'] if any(d['label']==f['label'] and iou(d['box'],f['box'])>=.5 for d in predictions)}
    assert found=={'with_mask','without_mask','mask_weared_incorrect'}


@pytest.mark.parametrize('face_count',[3,20])
def test_batch_classifier_preserves_multiple_faces(client,monkeypatch,face_count):
    # Exercise actual preprocessing, dynamic ONNX batch, and per-face mapping
    # with three controlled detector boxes. This checks plumbing, not accuracy.
    engine=app.state.engine
    class Detector:
        def setInputSize(self,size):self.size=size
        def setScoreThreshold(self,value):pass
        def detect(self,image):
            rows=[]
            for i in range(face_count):
                col,row=i%5,i//5
                rows.append([(col*.2+.01)*self.size[0],(row*.25+.01)*self.size[1],.17*self.size[0],.22*self.size[1],
                             *([0]*10),.99-i*.001])
            return 1,np.array(rows,np.float32)
    monkeypatch.setattr(engine,'detector',Detector())
    result=engine.predict(np.zeros((200,600,3),np.uint8))
    assert len(result['detections'])==face_count
    assert sum(result['counts'].values())==face_count
    for detection in result['detections']:
        assert abs(sum(detection['probabilities'].values())-1)<1e-5
        assert detection['box'][2]>detection['box'][0]
