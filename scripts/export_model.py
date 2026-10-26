"""Export a SINGLE portable bundle: ONNX CNN + YuNet face detector + metadata."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import zipfile
import numpy as np
import torch
import onnxruntime as ort
from scripts.train import make_model, LABELS, NAMES

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    model=make_model()
    checkpoint=torch.load(ROOT/'artifacts/checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    x=torch.randn(3,3,160,160)
    onnx=ROOT/'artifacts/classifier.onnx'
    torch.onnx.export(model,x,onnx,input_names=['images'],output_names=['logits'],
                      dynamic_axes={'images':{0:'faces'},'logits':{0:'faces'}},opset_version=17,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(onnx),sess_options=opts,providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=model(x).numpy()
    actual=session.run(None,{'images':x.numpy()})[0]
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
    for n in [1,8]:
        assert session.run(None,{'images':np.zeros((n,3,160,160),dtype=np.float32)})[0].shape==(n,3)
    files={'classifier.onnx':onnx.read_bytes(),
           'face_detector.onnx':(ROOT/'models/face_detection_yunet_2023mar.onnx').read_bytes()}
    metadata={'format_version':2,'architecture':'MobileNetV3-Small CNN','labels':LABELS,'display_names':NAMES,
              'input_size':160,'color':'RGB','scale':255.,'mean':[.485,.456,.406],'std':[.229,.224,.225],
              'crop_padding':.1,'detector':'OpenCV YuNet CNN','detector_input_size':640,'detector_threshold':.7,'best_epoch':checkpoint['epoch'],
              'evaluation':json.loads((ROOT/'artifacts/evaluation.json').read_text()),
              'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
              'source':json.loads((ROOT/'artifacts/yunet_source.json').read_text()),
              'classifier_pretraining':'https://pytorch.org/vision/stable/models/mobilenetv3.html'}
    dest=ROOT/'models/mask_detector.bundle'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
        z.writestr('metadata.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        z.writestr('NOTICE.txt','Educational mask recognition lab. CNN fine-tuned on FMD. OpenCV YuNet face detector and MobileNetV3 pretrained weights are credited in metadata and docs/SOURCES.md.\n')
        for license_path in (ROOT/'docs/licenses').glob('*.txt'):
            z.writestr('licenses/'+license_path.name,license_path.read_bytes())
    (ROOT/'artifacts/model_export.json').write_text(json.dumps({'bundle':'models/mask_detector.bundle','size_bytes':dest.stat().st_size,
        'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'onnx_max_abs_error':float(np.abs(actual-expected).max()),'dynamic_batches_verified':[1,3,8]},indent=2))
    print(dest, dest.stat().st_size,flush=True)


if __name__=='__main__':main()
