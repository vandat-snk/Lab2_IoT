"""Download the public FMD dataset and the pinned OpenCV YuNet detector."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASET = "hmnshudhmn24/face-mask-detection"


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IoT-Lab3-education"})
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            if not data:
                raise ValueError("Empty download")
            temp = path.with_suffix(path.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(path)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    target = ROOT / "data/raw"
    target.mkdir(parents=True, exist_ok=True)
    info_path = target / "provenance.json"
    if info_path.exists():
        revision = json.loads(info_path.read_text())["revision"]
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}", timeout=30) as r:
            revision = json.load(r)["sha"]
        info_path.write_text(json.dumps({
            "dataset": DATASET, "revision": revision,
            "original": "https://www.kaggle.com/datasets/andrewmvd/face-mask-detection",
            "mirror": f"https://huggingface.co/datasets/{DATASET}",
            "classes": ["with_mask", "without_mask", "mask_weared_incorrect"],
            "note": "Public FMD dataset by Andrewmvd, downloaded from a public mirror."
        }, indent=2), encoding="utf-8")
    tasks = []
    for i in range(853):
        for folder, ext in [("annotations", "xml"), ("images", "png")]:
            name = f"{folder}/maksssksksss{i}.{ext}"
            tasks.append((f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}", target / name))
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download, *args) for args in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 100 == 0:
                print(f"Dataset files: {i}/{len(tasks)}", flush=True)
    detector_revision='f12e12798e8314f7c074a6656816c048dcc95b7a'
    detector=ROOT/'models/face_detection_yunet_2023mar.onnx'
    download(f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{detector_revision}/models/face_detection_yunet/{detector.name}',detector)
    if hashlib.sha256(detector.read_bytes()).hexdigest()!='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4':
        raise ValueError('YuNet SHA256 mismatch')
    download(f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/README.md", target / "DATASET_CARD.md")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(target.rglob("*")) if p.suffix in [".png", ".xml"]}
    (target / "sha256.json").write_text(json.dumps(manifest, indent=2))
    print(f"Ready: {len(manifest)} verified data files", flush=True)


if __name__ == "__main__":
    main()
