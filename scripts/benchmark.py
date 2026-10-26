"""Benchmark a real endpoint. Browser-only capture/render latency is deliberately not fabricated."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import httpx


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--requests',type=int,default=30)
    p.add_argument('--output',default='artifacts/latency')
    args=p.parse_args()
    image=Path(args.image).read_bytes();rows=[]
    with httpx.Client(timeout=120) as client:
        client.get(args.url.rstrip('/')+'/api/health').raise_for_status()
        for i in range(args.requests+1):
            t=time.perf_counter()
            response=client.post(args.url.rstrip('/')+'/api/predict',files={'file':('test.jpg',image,'image/jpeg')})
            response.raise_for_status();result=response.json();rtt=(time.perf_counter()-t)*1000
            rows.append({'timestamp':datetime.now(timezone.utc).isoformat(),'warmup':i==0,'request_id':result['request_id'],
                'backend':args.url,'model_sha256':result['model_sha256'],'round_trip_ms':rtt,**result['timing'],
                'face_count':result['face_count'],**result['counts']})
            print(f'{i:02d}: inference={result["timing"]["inference_ms"]:.1f} ms; RTT={rtt:.1f} ms; faces={result["face_count"]}',flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stable=rows[1:]
    def summary(key):
        values=sorted(r[key] for r in stable)
        import math
        return {'mean':statistics.mean(values),'median':statistics.median(values),'p95':values[math.ceil(.95*len(values))-1],
                'min':min(values),'max':max(values)}
    result={'backend':args.url,'steady_state_requests':len(stable),'warmup_excluded':1,'image':str(args.image),
            'inference_ms':summary('inference_ms'),'round_trip_ms':summary('round_trip_ms'),
            'end_to_end_ms':None,'note':'CLI measures inference and round trip only. Capture-to-painted-result must be measured/exported in the browser UI.'}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
