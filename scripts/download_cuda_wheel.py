"""Resumable range download of the pinned official Windows CUDA wheel."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-win_amd64.whl'
SIZE=2862036321
SHA256='ad9183864acdd99fc5143d7ca9d3d2e7ddfc9a9600ff43217825d4e5e9855ccc'
CHUNK=16*1024*1024


def main():
    folder=ROOT/'.cache/cuda-download';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'torch-2.9.1+cu128-cp313-cp313-win_amd64.whl'
    def part(i):
        start=i*CHUNK;end=min(SIZE,start+CHUNK)-1;path=folder/f'{i:04d}.part'
        if path.exists() and path.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Unexpected range response')
                    data=response.read()
                if len(data)!=end-start+1:raise ValueError('Incomplete range')
                path.write_bytes(data);return
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
    n=(SIZE+CHUNK-1)//CHUNK
    with ThreadPoolExecutor(max_workers=24) as pool:
        for done,f in enumerate(as_completed([pool.submit(part,i) for i in range(n)]),1):
            f.result()
            if done%10==0:print(f'CUDA wheel: {done}/{n} chunks',flush=True)
    digest=hashlib.sha256()
    with dest.open('wb') as out:
        for i in range(n):
            data=(folder/f'{i:04d}.part').read_bytes();digest.update(data);out.write(data)
    if digest.hexdigest()!=SHA256:raise ValueError('CUDA wheel SHA256 mismatch')
    print(f'Verified {dest}',flush=True)


if __name__=='__main__':main()
