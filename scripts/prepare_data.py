"""Split source-image groups BEFORE cropping; never split faces from the same image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
from PIL import Image
from backend.geometry import padded_box

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["with_mask", "without_mask", "mask_weared_incorrect"]


def main():
    raw, out = ROOT / "data/raw", ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    sources, excluded = [], Counter()
    for xml in sorted((raw / "annotations").glob("*.xml")):
        tree = ET.parse(xml).getroot()
        filename = tree.findtext("filename")
        if filename != xml.stem + '.png':
            raise ValueError(f'Unexpected dataset filename in {xml.name}')
        path = raw / "images" / filename
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            thumb = np.asarray(im.resize((9, 8)).convert("L"))
            dh = sum(int(v) << i for i, v in enumerate((thumb[:, 1:] > thumb[:, :-1]).flat))
            exact = hashlib.sha256(im.tobytes()).hexdigest()
        faces = []
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name not in LABELS:
                raise ValueError(f"Unknown label {name}")
            box = [int(float(obj.findtext(f"bndbox/{k}"))) for k in ["xmin", "ymin", "xmax", "ymax"]]
            box = [max(0, box[0]-1), max(0, box[1]-1), min(w, box[2]), min(h, box[3])]
            if box[2]-box[0] < 12 or box[3]-box[1] < 12:
                excluded[name] += 1
                continue
            faces.append({"label": name, "box": box})
        if faces:
            sources.append({"source": path.name, "exact": exact, "dhash": dh, "faces": faces})
    parent = list(range(len(sources)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(sources):
        for j in range(i):
            b = sources[j]
            if a["exact"] == b["exact"] or (a["dhash"] ^ b["dhash"]).bit_count() <= 4:
                parent[find(i)] = find(j)
    groups = {}
    for i, src in enumerate(sources):
        src["group"] = find(i)
        groups.setdefault(find(i), []).append(i)
    ids = list(groups)
    total = Counter(f["label"] for s in sources for f in s["faces"])
    best, best_score = None, float("inf")
    rng = random.Random(42)
    for _ in range(1500):
        rng.shuffle(ids)
        n = len(ids)
        parts = [ids[:int(.70*n)], ids[int(.70*n):int(.85*n)], ids[int(.85*n):]]
        counts = [Counter(f["label"] for gid in part for i in groups[gid] for f in sources[i]["faces"]) for part in parts]
        if any(c[label] < 5 for c in counts for label in LABELS):
            continue
        score = sum(abs(c[label]/total[label]-ratio) for c, ratio in zip(counts, [.70,.15,.15]) for label in LABELS)
        if score < best_score:
            best, best_score = [list(p) for p in parts], score
    if best is None or any(total[label] < 20 for label in LABELS):
        raise ValueError("Not enough independent source data for three classes")
    manifest, source_manifest, stats = [], [], {}
    for split, group_ids in zip(["train", "val", "test"], best):
        count = Counter()
        for gid in group_ids:
            for idx in groups[gid]:
                src = sources[idx]
                source_manifest.append({k:v for k,v in src.items() if k != "dhash"} | {"split": split})
                with Image.open(raw / "images" / src["source"]) as im:
                    im = im.convert("RGB")
                    for fidx, face in enumerate(src["faces"]):
                        name = f"{Path(src['source']).stem}_{fidx}.jpg"
                        dest = out / split / face["label"] / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        im.crop(padded_box(face["box"], *im.size)).resize((160,160), Image.Resampling.BILINEAR).save(dest, quality=95)
                        manifest.append({"path": str(dest.relative_to(ROOT)).replace('\\','/'), "label": face["label"], "source":src["source"], "group":gid, "split":split})
                        count[face["label"]] += 1
        stats[split] = dict(count)
    report = {"seed":42,"classes":LABELS,"image_size":160,"padding":.1,"min_face_side":12,
              "split_method":"70/15/15 by source-image groups, RGB duplicates and dHash distance <=4 grouped before cropping",
              "source_images":len(sources), "source_groups":len(groups), "counts":stats,"excluded_tiny_faces":dict(excluded)}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out / "sources.json").write_text(json.dumps(source_manifest,indent=2),encoding="utf-8")
    (ROOT / "artifacts/source_split.json").write_text(json.dumps(source_manifest,indent=2),encoding="utf-8")
    (ROOT / "artifacts/dataset_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
