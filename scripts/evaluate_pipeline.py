"""Evaluate the detector + three-class CNN on held-out full source images."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
import json
import time
import cv2
from backend.inference import ModelEngine

ROOT=Path(__file__).resolve().parents[1]
LABELS=['with_mask','without_mask','mask_weared_incorrect']


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area else 0.


def annotate(image,result):
    output=image.copy()
    colors={'with_mask':(113,183,28),'without_mask':(70,65,225),'mask_weared_incorrect':(0,175,240),'uncertain':(160,160,160)}
    for face in result['detections']:
        x1,y1,x2,y2=face['box'];color=colors[face['label']]
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        cv2.putText(output,f'{face["label"]} {face["confidence"]:.2f}',(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return output


def main():
    engine=ModelEngine(ROOT/'models/mask_detector.bundle')
    sources=json.loads((ROOT/'data/processed/sources.json').read_text())
    sources=[s for s in sources if s['split']=='test']
    tp=Counter();fp=Counter();fn=Counter();matched=0;gt_total=0;pred_total=0;details=[];candidates=[]
    start=time.perf_counter()
    for index,source in enumerate(sources):
        image=cv2.imread(str(ROOT/'data/raw/images'/source['source']))
        result=engine.predict(image)
        truth=source['faces'];available=set(range(len(truth)));matched_labels=set();per_image=[]
        gt_total+=len(truth);pred_total+=result['face_count']
        for pred in sorted(result['detections'],key=lambda d:d['face_confidence'],reverse=True):
            scores=[(iou(pred['box'],truth[j]['box']),j) for j in available]
            score,j=max(scores,default=(0,None))
            if score>=.5:
                available.remove(j);matched+=1;actual=truth[j]['label']
                if pred['label']==actual:tp[actual]+=1;matched_labels.add(actual)
                else:fp[pred['label']]+=1;fn[actual]+=1
                per_image.append({'truth':actual,'predicted':pred['label'],'iou':score})
            else:fp[pred['label']]+=1
        for j in available:fn[truth[j]['label']]+=1
        details.append({'source':source['source'],'ground_truth_faces':len(truth),'predicted_faces':result['face_count'],
                        'correct_classes_found':sorted(matched_labels),'counts':result['counts'],'matches':per_image})
        if set(LABELS).issubset(matched_labels):candidates.append((source['source'],result,image))
        if index%30==0:print(f'Full pipeline: {index+1}/{len(sources)} images',flush=True)
    metrics={}
    for label in LABELS:
        precision=tp[label]/(tp[label]+fp[label]) if tp[label]+fp[label] else 0.
        recall=tp[label]/(tp[label]+fn[label]) if tp[label]+fn[label] else 0.
        metrics[label]={'tp':tp[label],'fp':fp[label],'fn':fn[label],'precision':precision,'recall':recall,
                       'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    report={'source_split':'held-out test source images','source_images':len(sources),'iou_threshold':.5,
        'detection_threshold':.7,'classification_threshold':.5,'ground_truth_faces':gt_total,'predicted_faces':pred_total,
        'detector_recall':matched/gt_total,'detector_precision':matched/pred_total if pred_total else 0.,
        'per_class_end_to_end':metrics,'macro_f1':sum(m['f1'] for m in metrics.values())/3,
        'three_class_natural_frames_found':len(candidates),'evaluation_seconds':time.perf_counter()-start,
        'detector':engine.meta['detector'],'detector_input_size':engine.meta['detector_input_size'],
        'note':'Greedy IoU matching on original full images. Ground truth excludes faces <12px, matching crop dataset protocol. False positives on excluded tiny faces may lower precision. This is not mAP. YuNet threshold selected on validation. Pipeline iterated after inspecting an SSD baseline on this test split, so final pipeline results are exploratory, not a completely untouched benchmark. CNN checkpoint selection uses validation only.'}
    (ROOT/'artifacts/pipeline_evaluation.json').write_text(json.dumps(report,indent=2))
    (ROOT/'artifacts/pipeline_predictions.json').write_text(json.dumps(details,indent=2))
    sample_dir=ROOT/'web/samples';sample_dir.mkdir(parents=True,exist_ok=True)
    demo_split='test'
    if not candidates:
        # Validation examples may be used for a clearly labelled demonstration,
        # never mixed into the held-out test metrics above.
        for source in json.loads((ROOT/'data/processed/sources.json').read_text()):
            if source['split']!='val' or len({f['label'] for f in source['faces']})!=3:continue
            image=cv2.imread(str(ROOT/'data/raw/images'/source['source']));result=engine.predict(image)
            hits={f['label'] for f in source['faces'] if any(d['label']==f['label'] and iou(d['box'],f['box'])>=.5 for d in result['detections'])}
            if set(LABELS).issubset(hits):candidates.append((source['source'],result,image))
        demo_split='val'
    if candidates:
        # This is selected as a demonstration, not used to estimate performance.
        name,result,image=max(candidates,key=lambda a:sum(d['confidence'] for d in a[1]['detections'])/max(1,a[1]['face_count']))
        cv2.imwrite(str(sample_dir/'three_classes.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
        jpeg=cv2.imread(str(sample_dir/'three_classes.jpg'));jpeg_result=engine.predict(jpeg)
        if not all(jpeg_result['counts'][label]>0 for label in LABELS):
            raise RuntimeError('JPEG demonstration did not retain all three classes. Select another source frame.')
        cv2.imwrite(str(ROOT/'artifacts/three_classes_detected.jpg'),annotate(jpeg,jpeg_result))
        (ROOT/'artifacts/demo_sample.json').write_text(json.dumps({'source':name,'split':demo_split,'type':'natural image, selected demonstration','result':jpeg_result},indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
