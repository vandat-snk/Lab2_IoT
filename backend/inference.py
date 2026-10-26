from pathlib import Path
import hashlib
import json
import threading
import time
import zipfile
import cv2
import numpy as np
import onnxruntime as ort
from .geometry import padded_box


class ModelEngine:
    def __init__(self, bundle_path):
        self.path = Path(bundle_path)
        with zipfile.ZipFile(self.path) as z:
            self.meta = json.loads(z.read('metadata.json'))
            if self.meta.get('format_version') != 2:
                raise ValueError('Unsupported model bundle format')
            files = {name:z.read(name) for name in ['classifier.onnx','face_detector.onnx']}
        for name, payload in files.items():
            if hashlib.sha256(payload).hexdigest() != self.meta['sha256'][name]:
                raise ValueError(f'Corrupt model file: {name}')
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        side=self.meta['detector_input_size']
        self.detector = cv2.FaceDetectorYN.create('onnx',np.frombuffer(files['face_detector.onnx'],np.uint8),
            np.empty(0,np.uint8),(side,side),self.meta['detector_threshold'],.3,5000,
            cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.classifier=ort.InferenceSession(files['classifier.onnx'],sess_options=opts,providers=['CPUExecutionProvider'])
        self.lock=threading.Lock()
        cv2.setNumThreads(1)
        # Warm-up is intentionally outside measured steady-state inference.
        self.detector.detect(np.zeros((side,side,3),np.uint8))
        self.classifier.run(None,{'images':np.zeros((1,3,self.meta['input_size'],self.meta['input_size']),np.float32)})

    def predict(self, image, detection_threshold=.7, classification_threshold=.5):
        wait_start=time.perf_counter()
        with self.lock:
            queue_ms=(time.perf_counter()-wait_start)*1000
            h,w=image.shape[:2]
            scale=self.meta['detector_input_size']/max(h,w)
            sw,sh=max(32,round(w*scale)),max(32,round(h*scale))
            resized=cv2.resize(image,(sw,sh))
            self.detector.setInputSize((sw,sh));self.detector.setScoreThreshold(detection_threshold)
            t=time.perf_counter();_,raw=self.detector.detect(resized);detector_ms=(time.perf_counter()-t)*1000
            boxes=[];scores=[]
            for row in ([] if raw is None else raw):
                if not np.isfinite(row).all() or float(row[-1]) < detection_threshold:
                    continue
                x1,y1,x2,y2=np.rint([row[0]*w/sw,row[1]*h/sh,(row[0]+row[2])*w/sw,(row[1]+row[3])*h/sh]).astype(int)
                x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
                if x2-x1<10 or y2-y1<10:continue
                boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1)]);scores.append(float(row[-1]))
            keep=cv2.dnn.NMSBoxes(boxes,scores,detection_threshold,.35) if boxes else []
            keep=np.asarray(keep,dtype=int).reshape(-1).tolist()
            keep=sorted(keep,key=lambda i:scores[i],reverse=True)[:100]
            crops=[];results=[]
            for i in keep:
                x,y,bw,bh=boxes[i]
                box=[x,y,x+bw,y+bh]
                x1,y1,x2,y2=padded_box(box,w,h,self.meta['crop_padding'])
                crop=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
                crop=cv2.resize(crop,(self.meta['input_size'],self.meta['input_size']),interpolation=cv2.INTER_LINEAR)
                arr=crop.astype(np.float32)/self.meta['scale']
                arr=(arr-np.asarray(self.meta['mean'],np.float32))/np.asarray(self.meta['std'],np.float32)
                crops.append(arr.transpose(2,0,1))
                results.append({'box':box,'face_confidence':round(scores[i],6)})
            classifier_ms=0.
            if crops:
                outputs=[]
                for offset in range(0,len(crops),16):
                    batch=np.ascontiguousarray(np.stack(crops[offset:offset+16]),dtype=np.float32)
                    t=time.perf_counter();outputs.append(self.classifier.run(None,{'images':batch})[0]);classifier_ms+=(time.perf_counter()-t)*1000
                logits=np.concatenate(outputs,axis=0)
                logits-=logits.max(axis=1,keepdims=True)
                probs=np.exp(logits);probs/=probs.sum(axis=1,keepdims=True)
                for result,p in zip(results,probs):
                    idx=int(np.argmax(p));confidence=float(p[idx]);known=confidence>=classification_threshold
                    result.update({'label':self.meta['labels'][idx] if known else 'uncertain',
                        'display_name':self.meta['display_names'][idx] if known else 'Chưa chắc chắn',
                        'confidence':round(confidence,6),'probabilities':dict(zip(self.meta['labels'],map(float,p)))})
        counts={label:sum(r['label']==label for r in results) for label in self.meta['labels']+['uncertain']}
        return {'detections':results,'counts':counts,'face_count':len(results),'image':{'width':w,'height':h},
                'timing':{'inference_ms':detector_ms+classifier_ms,'face_detection_ms':detector_ms,
                          'cnn_ms':classifier_ms,'queue_ms':queue_ms},'model_sha256':self.sha256}
