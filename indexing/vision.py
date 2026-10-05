"""One small client for the models the MVP uses: JSON answers, with or without an image.
Works with Gemini (GEMINI_API_KEY) and/or Groq (GROQ_API_KEY). Finds usable models by itself and
moves to the next one when a daily limit is hit."""
import base64, json, os, re, time
import requests
from dotenv import load_dotenv

load_dotenv()
GEM = "https://generativelanguage.googleapis.com/v1beta"
GROQ = "https://api.groq.com/openai/v1"
_cands = {}          # needs_image -> [(provider, model)]
_dead = set()
TINY = ("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/"
        "yQALCAABAAEBAREA/8wABgAQEAX/2gAIAQEAAD8A0s8g/9k=")


class NoModel(RuntimeError):
    pass


def _gemini_models():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return []
    r = requests.get(f"{GEM}/models", params={"key": key, "pageSize": 200}, timeout=20)
    r.raise_for_status()
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    names = [n for n in names if "flash" in n and not re.search(r"tts|image|live|audio|embed|exp|thinking|native", n)]
    # stable names first, then newest, lite before full (lite has the larger free daily allowance)
    names.sort(key=lambda n: ("preview" in n, [-float(x) for x in re.findall(r"\d+\.?\d*", n)[:1]] or [0], "lite" not in n))
    return [("gemini", n) for n in names[:6]]


def _groq_models(needs_image):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return []
    r = requests.get(f"{GROQ}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    if needs_image:
        ids = [i for i in ids if re.search(r"vision|scout|maverick|llava|pixtral|-vl|gemma-3|llama-4", i)]
    else:
        ids = [i for i in ids if re.search(r"gpt-oss|llama|kimi|qwen|gemma", i)
               and not re.search(r"guard|whisper|tts|safeguard|prompt", i)]
        ids.sort(key=lambda i: ("120b" not in i, "70b" not in i))
    return [("groq", i) for i in ids]


def candidates(needs_image):
    if needs_image not in _cands:
        _cands[needs_image] = _gemini_models() + _groq_models(needs_image)
    left = [c for c in _cands[needs_image] if c not in _dead]
    if not left:
        raise NoModel("No usable model" + (" that can read images" if needs_image else "") +
                      ". Add GEMINI_API_KEY to .env (free at aistudio.google.com), or the daily limits are used up.")
    return left


def _call(provider, model, prompt, image_b64, max_tokens):
    if provider == "gemini":
        parts = [{"text": prompt}] + ([{"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}] if image_b64 else [])
        return requests.post(f"{GEM}/models/{model}:generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                             json={"contents": [{"parts": parts}],
                                   "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens + 2500,   # room for model thinking
                                                        "responseMimeType": "application/json"}}, timeout=90)
    content = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    return requests.post(f"{GROQ}/chat/completions",
                         headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                         json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "user", "content": content if image_b64 else prompt}]}, timeout=90)


def _text(provider, r):
    j = r.json()
    if provider == "gemini":
        c = (j.get("candidates") or [{}])[0]
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            raise ValueError(f"empty answer (finishReason={c.get('finishReason')}, {str(j.get('promptFeedback', ''))[:80]})")
        return text
    return j["choices"][0]["message"]["content"]


def ask_json(prompt, image_bytes=None, max_tokens=700):
    """Returns (parsed JSON, model name). Raises NoModel when nothing is left, ValueError on a bad answer."""
    img = base64.b64encode(image_bytes).decode() if image_bytes else None
    short_waits = 0
    while True:
        provider, model = candidates(bool(img))[0]
        try:
            r = _call(provider, model, prompt, img, max_tokens)
        except requests.RequestException:
            short_waits += 1
            if short_waits > 4:
                raise ValueError("network error")
            time.sleep(3); continue
        if r.status_code == 429:
            m = re.search(r'retry in ([\d.]+)s|"retryDelay":\s*"(\d+)', r.text)
            wait = float(r.headers.get("retry-after") or (m and (m.group(1) or m.group(2))) or 8)
            daily = re.search(r"per day|PerDay|TPD|RPD|daily", r.text)
            if daily or wait > 70 or short_waits > 6:
                _dead.add((provider, model)); short_waits = 0
                print(f"  limit reached on {model}; switching", flush=True); continue
            short_waits += 1; time.sleep(wait + 1); continue
        if r.status_code in (400, 403, 404) and not img or r.status_code in (403, 404):
            _dead.add((provider, model)); continue          # model not available to this key
        if r.status_code == 400:
            if re.search(r"image|vision|multimodal|content must be a string", r.text, re.I):
                _dead.add((provider, model)); continue      # this model cannot read images
            raise ValueError(f"rejected: {r.text[:160]}")
        if r.status_code >= 500:
            short_waits += 1
            if short_waits > 4:
                _dead.add((provider, model)); short_waits = 0
            time.sleep(4); continue
        try:
            t = _text(provider, r).strip()
            t = re.sub(r"^```(?:json)?|```$", "", t).strip()
            return json.loads(t), model
        except Exception as e:
            raise ValueError(f"unparseable answer: {e}")


def probe(image_bytes):
    """Find a model that can really read an image. Returns its name, or raises NoModel."""
    while True:
        provider, model = candidates(True)[0]
        try:
            out, used = ask_json('What is the dominant colour of this image? Reply with JSON {"colour": "..."}.',
                                 image_bytes, max_tokens=200)
            return used
        except ValueError as e:
            print(f"  {model} did not work ({str(e)[:90]}); trying the next model", flush=True)
            _dead.add((provider, model))
