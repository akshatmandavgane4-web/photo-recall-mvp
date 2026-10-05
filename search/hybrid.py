"""Hybrid search over an indexed library: BM25 on captions/text/episode labels, optional embedding similarity,
and soft boosts for time, type and other clues. Nothing is ever hard-filtered, so a wrong clue cannot hide a photo."""
import json, math, re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = set("about a an the of in on at to for and or with from is was were are it its this that there their his her my our "
           "photo picture image pic showing shows shown displays displayed featuring features some one two".split())


def toks(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", str(s).lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("ies"): w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("ing"): w = w[:-3]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]
        out.append(w)
    return out


class Library:
    def __init__(self, name="seed-library"):
        d = ROOT / "data" / name
        self.dir = d
        self.thumbs = d / "thumbs"
        if not self.thumbs.exists() and (d / "thumbs.zip").exists():   # hosted copy ships thumbnails as one file
            import tempfile, zipfile
            try:
                zipfile.ZipFile(d / "thumbs.zip").extractall(self.thumbs)
            except OSError:
                self.thumbs = Path(tempfile.mkdtemp()); zipfile.ZipFile(d / "thumbs.zip").extractall(self.thumbs)
        skip = set(json.load(open(d / "exclude.json"))) if (d / "exclude.json").exists() else set()
        self.photos = [r for r in (json.loads(l) for l in open(d / "index.jsonl", encoding="utf-8")) if r["file"] not in skip]
        eps = json.load(open(d / "episodes.json", encoding="utf-8")) if (d / "episodes.json").exists() else []
        self.episodes = {e["episode_id"]: e for e in eps}
        self.by_id = {p["photo_id"]: p for p in self.photos}
        for p in self.photos:
            e = self.episodes.get(p.get("episode_id"), {})
            p["_ep"] = "" if e.get("ungrouped") else f"{e.get('label', '')} {e.get('summary', '')}"
            p["_main"] = toks(f"{p['caption']} {' '.join(p['objects'])} {p['setting']}")
            p["_ocr"] = toks(p["ocr_text"])
            p["_ept"] = toks(re.sub(r"\d", " ", p["_ep"]))
            p["_all"] = p["_main"] + p["_ocr"] + p["_ept"]
            p["_dt"] = datetime.fromisoformat(p["taken_at"])
        n = len(self.photos)
        df = Counter(w for p in self.photos for w in set(p["_all"]))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.avg = sum(len(p["_all"]) for p in self.photos) / max(n, 1)
        self.vecs = None
        try:                                             # optional: embeddings built by indexing/embed.py
            import numpy as np
            v = np.load(d / "vectors.npy"); ids = json.load(open(d / "vector_ids.json"))
            self.vecs = {i: v[k] for k, i in enumerate(ids)}
        except Exception:
            pass

    def thumb(self, pid):
        return str(self.thumbs / f"{pid}.jpg")

    def _bm25(self, p, weighted_terms):
        s, hits = 0.0, []
        tf_main, tf_ocr, tf_ep = Counter(p["_main"]), Counter(p["_ocr"]), Counter(p["_ept"])
        L = len(p["_all"]) / self.avg if self.avg else 1
        for w, (wt, field) in weighted_terms.items():
            tf = tf_main[w] + 0.6 * tf_ocr[w] + 1.2 * tf_ep[w] if field == "any" else 2.0 * tf_ocr[w] + 0.3 * tf_main[w]
            if tf:
                s += wt * self.idf.get(w, 0) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * L))
                if wt >= 1:
                    hits.append(w)
        return s, hits

    def search(self, clues, rejected=(), near=None, qvec=None, k=60):
        """clues: merged clue state (see agent/core.py). Returns ranked [(photo, score, reasons)]."""
        terms = {}
        def add(words, wt, field="any"):
            for w in toks(" ".join(words) if isinstance(words, list) else words or ""):
                if wt > terms.get(w, (0, ""))[0]:
                    terms[w] = (wt, field)
        add(clues.get("expanded", []), 0.45)
        add(clues.get("visual", []), 0.8)
        add(clues.get("place", ""), 1.0)
        add(clues.get("episode_hint", ""), 1.0)
        add(clues.get("semantic", []), 1.0)
        add(clues.get("text_in_image", []), 1.6, "ocr")
        neg = set(toks(" ".join(clues.get("negatives", []))))
        t = clues.get("time") or {}
        lo = datetime.strptime(t["from"], "%Y-%m") if t.get("from") else None
        hi = datetime.strptime(t["to"], "%Y-%m") if t.get("to") else None
        tconf = float(t.get("confidence", 0.5)) if (lo or hi) else 0
        near_p = self.by_id.get(near) if near else None
        raw = []
        for p in self.photos:
            s, hits = self._bm25(p, terms)
            raw.append([p, s, hits])
        top = max((r[1] for r in raw), default=0) or 1
        out = []
        for p, s, hits in raw:
            score, why = s / top, []
            if hits:
                why.append("matches " + ", ".join(f"“{h}”" for h in list(dict.fromkeys(hits))[:4]))
            if qvec is not None and self.vecs is not None and p["photo_id"] in self.vecs:
                score = 0.5 * score + 0.5 * max(0.0, float(self.vecs[p["photo_id"]] @ qvec))
            if tconf and p["date_confidence"] == "high":     # never penalize photos without a trustworthy date
                inside = (not lo or p["_dt"] >= lo) and (not hi or p["_dt"].replace(day=1) <= hi)
                if inside:
                    score += 0.3 * tconf; why.append("taken " + p["_dt"].strftime("%b %Y"))
                else:
                    gap = min(abs((p["_dt"] - x).days) for x in (lo, hi) if x) / 365
                    score -= min(0.25, 0.12 * gap) * tconf
            if clues.get("photo_type") and p["type"] == clues["photo_type"]:
                score += 0.2; why.append(p["type"])
            if clues.get("indoor") is not None and p["indoor"] is not None:
                score += 0.12 if p["indoor"] == clues["indoor"] else -0.08
            if clues.get("people") and p["people_count"] is not None:
                want = {"none": p["people_count"] == 0, "alone": p["people_count"] == 1, "group": p["people_count"] >= 2}
                score += 0.12 if want.get(clues["people"]) else -0.08
            if clues.get("has_text") is not None:
                score += 0.1 if bool(p["ocr_text"]) == clues["has_text"] else -0.06
            if neg and neg & set(p["_main"]):
                score -= 0.3
            if near_p and p is not near_p:
                if p.get("episode_id") == near_p.get("episode_id") and "ungrouped" not in str(p.get("episode_id")):
                    score += 0.45; why.append("same occasion as your near-miss")
                elif near_p["date_confidence"] == p["date_confidence"] == "high" and abs((p["_dt"] - near_p["_dt"]).days) <= 3:
                    score += 0.25; why.append("taken around the same days")
                shared = set(p["_main"]) & set(near_p["_main"])
                score += min(0.3, 0.03 * len(shared))
                if self.vecs is not None and p["photo_id"] in self.vecs and near in self.vecs:
                    score += 0.3 * max(0.0, float(self.vecs[p["photo_id"]] @ self.vecs[near]))
            if p["photo_id"] in rejected:
                score -= 2
            out.append((p, score, why))
        out.sort(key=lambda x: -x[1])
        return out[:k]
