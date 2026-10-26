from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
from pathlib import Path
import time
from uuid import UUID, uuid4
import warnings
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .database import save_measurement
from .inference import ModelEngine

ROOT=Path(__file__).resolve().parents[1]
MAX_BYTES=8*1024*1024
Image.MAX_IMAGE_PIXELS=16_000_000
LOG=logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    app.state.engine=None
    bundle=Path(os.getenv('MODEL_PATH',str(ROOT/'models/mask_detector.bundle')))
    if bundle.is_file():
        app.state.engine=ModelEngine(bundle)
    else:
        LOG.warning('Model bundle missing. Train and export first.')
    yield


app=FastAPI(title='IoT Lab 3 • CNN nhận dạng khẩu trang',version='1.0.0',lifespan=lifespan)
origins=[s.strip() for s in os.getenv('CORS_ORIGINS','http://localhost:8000,http://127.0.0.1:8000,http://localhost:7860,http://127.0.0.1:7860').split(',') if s.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'],expose_headers=['Server-Timing','X-Request-ID'])


@app.get('/api/health')
def health():
    engine=app.state.engine
    if engine is None:raise HTTPException(503,'Chưa có model. Chạy huấn luyện và xuất model trước.')
    return {'status':'ok','model_sha256':engine.sha256,'labels':engine.meta['labels'],
            'architecture':engine.meta['architecture'],'execution_provider':'CPUExecutionProvider',
            'database_configured':bool(os.getenv('DATABASE_URL'))}


@app.get('/api/model-info')
def model_info():
    if app.state.engine is None:raise HTTPException(503,'Model unavailable')
    return app.state.engine.meta


@app.get('/api/model/download')
def download_model():
    if app.state.engine is None:raise HTTPException(503,'Model unavailable')
    return FileResponse(app.state.engine.path,media_type='application/octet-stream',filename='mask_detector.bundle')


@app.post('/api/predict')
def predict(file:UploadFile=File(...),detection_threshold:float=Query(.7,ge=.1,le=.95),
            classification_threshold:float=Query(.5,ge=0,le=.99)):
    started=time.perf_counter()
    if app.state.engine is None:raise HTTPException(503,'Model unavailable')
    payload=file.file.read(MAX_BYTES+1)
    if len(payload)>MAX_BYTES:raise HTTPException(413,'Ảnh tối đa 8 MB')
    if not payload:raise HTTPException(400,'Ảnh trống')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as check:
                if check.width*check.height>16_000_000 or max(check.size)>4096:
                    raise HTTPException(413,'Ảnh tối đa 4096 pixel mỗi chiều và 16 megapixel')
                if check.format not in ['JPEG','PNG','WEBP']:raise HTTPException(415,'Chỉ nhận JPEG, PNG hoặc WebP')
                check.verify()
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Invalid image')
    except HTTPException:raise
    except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise HTTPException(400,'Không đọc được ảnh')
    decode_ms=(time.perf_counter()-started)*1000
    result=app.state.engine.predict(image,detection_threshold,classification_threshold)
    result['request_id']=str(uuid4())
    result['timing']['decode_ms']=decode_ms
    result['timing']['server_processing_ms']=(time.perf_counter()-started)*1000
    result['input_bytes']=len(payload)
    return result


class Measurement(BaseModel):
    request_id:UUID
    model_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    inference_ms:float=Field(ge=0,le=300000,allow_inf_nan=False)
    round_trip_ms:float=Field(ge=0,le=300000,allow_inf_nan=False)
    end_to_end_ms:float=Field(ge=0,le=300000,allow_inf_nan=False)
    counts:dict[str,int]


@app.post('/api/measurements')
def measurements(body:Measurement,background:BackgroundTasks):
    expected={'with_mask','without_mask','mask_weared_incorrect','uncertain'}
    if set(body.counts)!=expected or any(n<0 or n>100 for n in body.counts.values()):
        raise HTTPException(422,'Invalid counts')
    if not os.getenv('DATABASE_URL'):return {'accepted':False,'reason':'PostgreSQL chưa cấu hình; vẫn có thể xuất CSV trên trình duyệt.'}
    record=body.model_dump(mode='json');background.add_task(save_measurement,record)
    return {'accepted':True,'detail':'Đã nhận để ghi PostgreSQL; accepted không xác nhận ghi thành công.'}


@app.get('/')
def index():return FileResponse(ROOT/'web/index.html')


app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
