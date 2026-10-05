"""Photo Recall MVP: describe what you remember, steer from near-misses. Run: streamlit run web/app.py (from mvp/)"""
import json, os, sys, time, uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
st.set_page_config(page_title="Photo Recall", layout="wide")
try:
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
except Exception:
    pass
from search.hybrid import Library          # noqa: E402
from agent import core                     # noqa: E402

PAGE, MAX_QUESTIONS, MAX_PER_EPISODE = 12, 3, 4
EXAMPLES = ["the day at the water park", "a lecture slide about bending metal", "a menu I photographed at a café",
            "friends at a palace, a couple of years ago", "a flight search I had open on my laptop"]


@st.cache_resource
def library():
    return Library("seed-library")


L = library()
S = st.session_state


def log(kind, **payload):
    """Events map onto the funnel. Logging must never break the flow, and never stores captions or raw text."""
    try:
        ev = {"event_id": uuid.uuid4().hex[:12], "session_id": S.sid, "attempt": S.attempt, "ts": datetime.now().isoformat(timespec="seconds"),
              "type": kind, "payload": payload}
        S.events.append(ev)
        (ROOT / "data").mkdir(exist_ok=True)
        with open(ROOT / "data" / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception:
        pass


def reset(abandon=True):
    if abandon and S.get("turns") and not S.get("found"):
        log("attempt_abandoned", turn=S.turns, reason="started_over")
    S.update(clues=dict(core.EMPTY), rejected=set(), near=None, turns=0, asked=set(), questions=0, question=None,
             found=None, notes=[], results=[], t0=None, misses=0, more=0, attempt=S.get("attempt", 0) + 1, gave_up=False)


if "sid" not in S:
    S.sid, S.events, S.attempt = uuid.uuid4().hex[:8], [], 0
    reset(abandon=False)


def run_search(note=None):
    ranked = L.search(S.clues, rejected=S.rejected, near=S.near)
    S.pool = [p for p, _, _ in ranked[:40] if p["photo_id"] not in S.rejected]
    per_ep, page = {}, []
    for p, s, why in ranked:                       # keep one occasion from crowding out the others
        e = p.get("episode_id")
        if per_ep.get(e, 0) >= MAX_PER_EPISODE and len(page) < PAGE:
            continue
        per_ep[e] = per_ep.get(e, 0) + 1
        page.append((p, s, why))
        if len(page) >= PAGE + S.more * PAGE:
            break
    S.results = page
    if note:
        S.notes.append(note)
    log("results_shown", photo_ids=[p["photo_id"] for p, _, _ in page], turn=S.turns)


def submit(text):
    text = (text or "").strip()[:400]
    if not text:
        return
    delta, meta = core.parse(text, S.clues)
    S.notes = []
    if meta["intent"] == "vague":
        S.notes.append("What do you remember about it: where you were, who was there, what was in it, or what it was for?")
        return
    if meta["intent"] == "other":
        S.notes.append("I can only help you find a photo in this library. Tell me what you remember about the one you want.")
        return
    if S.turns == 0:
        S.t0 = time.time(); log("attempt_started")
    S.turns += 1
    before = set(core.clue_types(S.clues))
    S.clues = core.merge(S.clues, delta)
    S.more = 0
    log("clues_submitted", clue_types=core.clue_types(S.clues), new_types=sorted(set(core.clue_types(S.clues)) - before),
        count=len(core.chips(S.clues)), turn=S.turns, degraded=bool(meta.get("degraded")))
    if meta.get("people_named"):
        S.notes.append("I can't recognize specific people yet, so I'm using the rest of what you said. Anything else you remember?")
    if meta.get("unresolved_anchor"):
        S.notes.append(f"Roughly when was “{meta['unresolved_anchor']}”? A year or season is enough.")
    if meta.get("degraded"):
        S.notes.append("The language model is busy, so I searched on your exact words.")
    if S.question:                                  # they typed instead of answering: treat it as a new clue
        S.question = None
    run_search()


def not_here():
    S.rejected |= {p["photo_id"] for p, _, _ in S.results}
    S.misses += 1; S.more = 0; S.notes = []
    if S.misses >= 2 and S.misses % 2 == 0:         # a confident detail may be wrong: loosen it and say so
        S.clues, msg = core.relax(S.clues)
        if msg:
            S.notes.append(msg)
    q = core.next_question(S.pool, S.clues, S.asked) if S.questions < MAX_QUESTIONS else None
    if q:
        S.question = q; S.asked.add(q[0]); S.questions += 1
        log("question_asked", question_type=q[0], turn=S.turns)
    elif S.questions >= MAX_QUESTIONS or S.misses >= 5:
        S.gave_up = True
        log("not_in_library", turn=S.turns, misses=S.misses)
    run_search()


def answer(delta, label):
    key = S.question[0]
    S.question = None; S.turns += 1
    if delta:
        S.clues = core.merge(S.clues, delta)
    log("question_answered", question_type=key, answered=bool(delta), turn=S.turns)
    run_search()


def close_like(pid):
    S.near = pid; S.turns += 1; S.more = 0
    S.rejected |= {p["photo_id"] for p, _, _ in S.results if p["photo_id"] != pid}
    p = L.by_id[pid]
    e = L.episodes.get(p.get("episode_id"), {})
    S.notes = ["Looking for photos like that one" + (f", from “{e['label']}”" if e and not e.get("ungrouped") else "") + "."]
    log("near_miss_used", photo_id=pid, turn=S.turns)
    run_search()


def confirm(pid):
    rank = next((i + 1 for i, (p, _, _) in enumerate(S.results) if p["photo_id"] == pid), None)
    S.found = pid
    log("photo_confirmed", photo_id=pid, rank=rank, turn=S.turns, elapsed_s=round(time.time() - (S.t0 or time.time())),
        questions=S.questions, used_near_miss=S.near is not None)


# ------------------------------------------------------------------ page
st.title("Photo Recall")
st.caption("Find a photo from what you remember about it, even if that isn't much. This demo searches a sample library "
           f"of {len(L.photos)} photos; it is not connected to any Google Photos account.")

if S.found:
    p = L.by_id[S.found]
    st.success(f"Found it in {S.turns} step{'s' * (S.turns != 1)}.")
    c1, c2 = st.columns([1, 2])
    c1.image(L.thumb(S.found))
    c2.write(p["caption"]); c2.caption(core.explain(p, [], L.episodes) + " · " + p["_dt"].strftime("%-d %b %Y") * (p["date_confidence"] == "high"))
    b1, b2 = c2.columns(2)
    if b1.button("Find another photo", type="primary"):
        reset(abandon=False); st.rerun()
    if b2.button("That's not it after all"):
        log("confirmation_undone", photo_id=S.found); S.rejected.add(S.found); S.found = None; run_search(); st.rerun()
    st.stop()

with st.form("say", clear_on_submit=True):
    c1, c2 = st.columns([6, 1])
    text = c1.text_input("What do you remember?", max_chars=400, label_visibility="collapsed",
                         placeholder="Describe it in your own words: what was in it, the occasion, roughly when, where you were..."
                         if not S.turns else "Add anything else you remember...")
    go = c2.form_submit_button("Search" if not S.turns else "Add clue", type="primary", width="stretch")
if go and text.strip():
    submit(text); st.rerun()

if not S.turns:
    st.write("Try one of these, or type your own:")
    cols = st.columns(len(EXAMPLES))
    for c, e in zip(cols, EXAMPLES):
        if c.button(e, width="stretch"):
            submit(e); st.rerun()

chips = core.chips(S.clues)
if chips:
    st.caption("What I understood (tap to remove a clue):")
    cols = st.columns(min(len(chips), 6) + 1)
    for i, (key, idx, label) in enumerate(chips[:12]):
        if cols[i % 6].button(f"✕ {label}", key=f"chip{key}{idx}"):
            S.clues = core.remove_chip(S.clues, key, idx); run_search(); st.rerun()
    if cols[-1].button("Start over"):
        reset(); st.rerun()

for n in S.notes:
    st.info(n)

if S.question:
    key, qtext, options = S.question
    st.subheader(qtext)
    cols = st.columns(len(options) + 1)
    for c, (label, delta) in zip(cols, options):
        if c.button(label, key=f"q{label}", width="stretch"):
            answer(delta, label); st.rerun()
    if cols[-1].button("Not sure, skip", width="stretch"):
        answer(None, "skip"); st.rerun()

if S.gave_up:
    st.warning("I may not have this photo: it might not be in this library. You can browse by occasion below, "
               "add another detail, or start over.")
    with st.expander("Browse by occasion"):
        for e in sorted(L.episodes.values(), key=lambda e: e["start_at"], reverse=True):
            ids = [i for i in e["photo_ids"] if i in L.by_id]
            if len(ids) >= 2 and st.button(f"{e['label']} ({len(ids)})", key="ep" + e["episode_id"]):
                S.clues = core.merge(dict(core.EMPTY), {"episode_hint": e["label"]}); S.rejected = set(); S.gave_up = False
                S.near = ids[0]; run_search(); st.rerun()

if S.results:
    groups = OrderedDict()
    for p, s, why in S.results:
        e = L.episodes.get(p.get("episode_id"), {})
        label = e.get("label", "Other photos") if len([1 for q, _, _ in S.results if q.get("episode_id") == p.get("episode_id")]) > 1 else "Single photos"
        groups.setdefault(label, []).append((p, why))
    if "Single photos" in groups:
        groups.move_to_end("Single photos")
    for label, items in groups.items():
        st.markdown(f"**{label}**")
        cols = st.columns(4)
        for i, (p, why) in enumerate(items):
            with cols[i % 4]:
                st.image(L.thumb(p["photo_id"]), width="stretch")
                st.caption(core.explain(p, [w for w in why if not w.startswith("from:")], {}) +
                           (" · " + p["_dt"].strftime("%b %Y") if p["date_confidence"] == "high" else " · date unknown"))
                b1, b2 = st.columns(2)
                if b1.button("This is it", key="y" + p["photo_id"], type="primary", width="stretch"):
                    confirm(p["photo_id"]); st.rerun()
                if b2.button("Close, like this", key="c" + p["photo_id"], width="stretch"):
                    close_like(p["photo_id"]); st.rerun()
    st.divider()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("None of these", width="stretch"):
        not_here(); st.rerun()
    if c2.button("Show more like these", width="stretch"):
        S.more += 1; run_search(); st.rerun()

if st.query_params.get("debug"):
    with st.expander("Session events"):
        st.download_button("Download events", "\n".join(json.dumps(e) for e in S.events), f"events_{S.sid}.jsonl")
        st.json(S.events[-12:])
