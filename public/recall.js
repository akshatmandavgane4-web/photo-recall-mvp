/* Photo Recall: search and conversation logic. Runs in the browser; also loadable in Node for tests.
   Mirrors search/hybrid.py and agent/core.py. Nothing is ever hard-filtered, so a wrong clue cannot hide a photo. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Recall = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var STOP = {};
  ("about a an the of in on at to for and or with from is was were are it its this that there their his her my our " +
    "photo picture image pic showing shows shown displays displayed featuring features some one two").split(" ")
    .forEach(function (w) { STOP[w] = 1; });
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function toks(s) {
    var out = [], m = String(s == null ? "" : s).toLowerCase().match(/[a-z0-9]+/g) || [];
    for (var i = 0; i < m.length; i++) {
      var w = m[i];
      if (STOP[w] || w.length < 2) continue;
      if (w.length > 4 && /ies$/.test(w)) w = w.slice(0, -3) + "y";
      else if (w.length > 4 && /ing$/.test(w)) w = w.slice(0, -3);
      else if (w.length > 3 && /s$/.test(w) && !/ss$/.test(w)) w = w.slice(0, -1);
      out.push(w);
    }
    return out;
  }
  function count(list) { var c = {}; list.forEach(function (w) { c[w] = (c[w] || 0) + 1; }); return c; }
  function ms(iso) { return Date.parse(iso + (iso.length <= 7 ? "-01T00:00:00Z" : "Z")); }
  function monthYear(t) { var d = new Date(t); return MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); }
  function fullDate(t) { var d = new Date(t); return d.getUTCDate() + " " + monthYear(t); }
  var DAY = 86400000;

  function Library(data) {
    var self = this;
    this.photos = data.photos;
    this.episodes = {};
    data.episodes.forEach(function (e) { self.episodes[e.episode_id] = e; });
    this.byId = {};
    this.photos.forEach(function (p) {
      self.byId[p.photo_id] = p;
      var e = self.episodes[p.episode_id] || {};
      var ep = e.ungrouped ? "" : (e.label || "") + " " + (e.summary || "");
      p._main = toks(p.caption + " " + (p.objects || []).join(" ") + " " + p.setting);
      p._ocr = toks(p.ocr_text);
      p._ept = toks(ep.replace(/\d/g, " "));
      p._all = p._main.concat(p._ocr, p._ept);
      p._tf = [count(p._main), count(p._ocr), count(p._ept)];
      p._set = count(p._main);
      p._dt = ms(p.taken_at);
    });
    var n = this.photos.length, df = {};
    this.photos.forEach(function (p) { Object.keys(count(p._all)).forEach(function (w) { df[w] = (df[w] || 0) + 1; }); });
    this.idf = {};
    Object.keys(df).forEach(function (w) { self.idf[w] = Math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)); });
    this.avg = this.photos.reduce(function (a, p) { return a + p._all.length; }, 0) / Math.max(n, 1);
  }

  Library.prototype.search = function (clues, rejected, near, k) {
    var self = this, terms = {};
    rejected = rejected || {};
    function add(words, wt, field) {
      toks(Array.isArray(words) ? words.join(" ") : words || "").forEach(function (w) {
        if (wt > (terms[w] ? terms[w][0] : 0)) terms[w] = [wt, field || "any"];
      });
    }
    add(clues.expanded, 0.45); add(clues.visual, 0.8); add(clues.place, 1.0); add(clues.episode_hint, 1.0);
    add(clues.semantic, 1.0); add(clues.text_in_image, 1.6, "ocr");
    var neg = toks((clues.negatives || []).join(" "));
    var t = clues.time || {}, lo = t.from ? ms(t.from) : null, hi = t.to ? ms(t.to) : null;
    var tconf = lo !== null || hi !== null ? Number(t.confidence == null ? 0.5 : t.confidence) : 0;
    var nearP = near ? this.byId[near] : null, words = Object.keys(terms);
    var raw = this.photos.map(function (p) {
      var s = 0, hits = [], L = self.avg ? p._all.length / self.avg : 1;
      words.forEach(function (w) {
        var wt = terms[w][0], a = p._tf[0][w] || 0, b = p._tf[1][w] || 0, c = p._tf[2][w] || 0;
        var tf = terms[w][1] === "any" ? a + 0.6 * b + 1.2 * c : 2.0 * b + 0.3 * a;
        if (tf) {
          s += wt * (self.idf[w] || 0) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * L));
          if (wt >= 1 && hits.indexOf(w) < 0) hits.push(w);
        }
      });
      return [p, s, hits];
    });
    var top = raw.reduce(function (m, r) { return Math.max(m, r[1]); }, 0) || 1;
    var out = raw.map(function (r) {
      var p = r[0], score = r[1] / top, why = [];
      if (r[2].length) why.push("Matched: " + r[2].slice(0, 4).join(", "));
      if (tconf && p.date_confidence === "high") {       // never penalize photos without a trustworthy date
        var d = new Date(p._dt), first = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
        if ((lo === null || p._dt >= lo) && (hi === null || first <= hi)) { score += 0.3 * tconf; why.push(monthYear(p._dt)); }
        else {
          var gaps = [lo, hi].filter(function (x) { return x !== null; }).map(function (x) { return Math.abs(Math.trunc((p._dt - x) / DAY)); });
          score -= Math.min(0.25, 0.12 * Math.min.apply(null, gaps) / 365) * tconf;
        }
      }
      if (clues.photo_type && p.type === clues.photo_type) { score += 0.2; why.push(p.type); }
      if (clues.indoor != null && p.indoor != null) score += p.indoor === clues.indoor ? 0.12 : -0.08;
      if (clues.people && p.people_count != null) {
        var want = { none: p.people_count === 0, alone: p.people_count === 1, group: p.people_count >= 2 };
        score += want[clues.people] ? 0.12 : -0.08;
      }
      if (clues.has_text != null) score += !!p.ocr_text === clues.has_text ? 0.1 : -0.06;
      if (neg.some(function (w) { return p._set[w]; })) score -= 0.3;
      if (nearP && p !== nearP) {
        if (p.episode_id === nearP.episode_id && String(p.episode_id).indexOf("ungrouped") < 0) { score += 0.45; why.push("same occasion as your pick"); }
        else if (nearP.date_confidence === "high" && p.date_confidence === "high" && Math.abs(Math.trunc((p._dt - nearP._dt) / DAY)) <= 3) { score += 0.25; why.push("taken around the same days"); }
        var shared = Object.keys(p._set).filter(function (w) { return nearP._set[w]; }).length;
        score += Math.min(0.3, 0.03 * shared);
      }
      if (rejected[p.photo_id]) score -= 2;
      return { photo: p, score: score, why: why };
    });
    out.sort(function (a, b) { return b.score - a.score; });
    return out.slice(0, k || 60);
  };

  // ---------------------------------------------------------------- clue state
  var EMPTY = { semantic: [], expanded: [], visual: [], text_in_image: [], negatives: [], episode_hint: "", place: "",
    time: null, photo_type: null, people: null, indoor: null, has_text: null };
  var CLUE_TYPES = { semantic: "semantic_content", visual: "visual_attributes", text_in_image: "text_in_image",
    episode_hint: "episodic_context", place: "approx_place", time: "relative_time", photo_type: "source_purpose", people: "people" };
  var VAGUE = /^\W*((that|the|this|a|my|some|one|photo|picture|pic|image|thing|from|before|earlier|old|please|find|me|i|want|need|looking|for|it)\W*)+$/i;

  function empty() { return JSON.parse(JSON.stringify(EMPTY)); }
  function has(v) { return Array.isArray(v) ? v.length > 0 : v !== null && v !== undefined && v !== ""; }
  function merge(state, delta) {
    var s = JSON.parse(JSON.stringify(state));
    Object.keys(delta || {}).forEach(function (k) {
      var v = delta[k];
      if (Array.isArray(v)) {
        var seen = {}, list = [];
        (s[k] || []).concat(v).forEach(function (x) { if (!seen[x]) { seen[x] = 1; list.push(x); } });
        s[k] = list.slice(0, 20);
      } else s[k] = v;
    });
    return s;
  }
  function clueTypes(state) { return Object.keys(CLUE_TYPES).filter(function (k) { return has(state[k]); }).map(function (k) { return CLUE_TYPES[k]; }).sort(); }
  function isVague(text, state) { return VAGUE.test(text) && !has(state.semantic) && !has(state.episode_hint) && !has(state.place); }

  function timeLabel(t) {
    function part(ym) { var a = ym.split("-"); return MONTHS[+a[1] - 1] + " " + a[0]; }
    if (t.from && t.to) {
      var f = t.from.split("-"), o = t.to.split("-");
      if (f[0] === o[0] && f[1] === "01" && o[1] === "12") return "around " + f[0];
      if (t.from === t.to) return "around " + part(t.from);
      return part(t.from) + " to " + part(t.to);
    }
    return t.from ? "after " + part(t.from) : "before " + part(t.to);
  }
  var TYPE_LABEL = { photo: "camera photo", screenshot: "screenshot", document: "photo of a document", receipt: "bill or receipt" };
  function chips(state) {
    var out = [];
    [["semantic", "What"], ["visual", "Look"], ["text_in_image", "Text"], ["negatives", "Not"]].forEach(function (x) {
      (state[x[0]] || []).forEach(function (v, i) { out.push({ key: x[0], idx: i, label: x[1], value: v }); });
    });
    if (state.episode_hint) out.push({ key: "episode_hint", idx: null, label: "Occasion", value: state.episode_hint });
    if (state.place) out.push({ key: "place", idx: null, label: "Where", value: state.place });
    if (state.time) out.push({ key: "time", idx: null, label: "When", value: timeLabel(state.time) });
    if (state.photo_type) out.push({ key: "photo_type", idx: null, label: "Type", value: TYPE_LABEL[state.photo_type] || state.photo_type });
    if (state.people) out.push({ key: "people", idx: null, label: "People", value: { none: "no people", alone: "one person", group: "several people" }[state.people] });
    if (state.indoor != null) out.push({ key: "indoor", idx: null, label: "Setting", value: state.indoor ? "indoors" : "outdoors" });
    if (state.has_text != null) out.push({ key: "has_text", idx: null, label: "Writing", value: state.has_text ? "has writing" : "no writing" });
    return out;
  }
  function removeChip(state, key, idx) {
    var s = merge(state, {});
    if (idx === null || idx === undefined) s[key] = EMPTY[key];
    else s[key] = s[key].filter(function (_, i) { return i !== idx; });
    if (key === "semantic" && !s.semantic.length) s.expanded = [];
    return s;
  }

  /* Used when the language model cannot be reached: search on the person's own words, and read a plain year if given. */
  function localParse(text) {
    var delta = {}, m = text.match(/\b(20[0-3]\d)\b/);
    if (m) delta.time = { from: m[1] + "-01", to: m[1] + "-12", confidence: 0.6 };
    var rest = text.replace(/\b(in|around|from|during)?\s*20[0-3]\d\b/g, " ").trim();
    if (rest) delta.semantic = [rest.slice(0, 80)];
    return delta;
  }

  var QUESTIONS = {   // only things a person can recognize, never exact dates
    indoor: { text: "Was it taken indoors or outdoors?", options: [["Indoors", { indoor: true }], ["Outdoors", { indoor: false }]],
      fn: function (p) { return p.indoor; } },
    people: { text: "Were there people in it?", options: [["No people", { people: "none" }], ["One person", { people: "alone" }], ["Several people", { people: "group" }]],
      fn: function (p) { return p.people_count == null ? null : Math.min(p.people_count, 2); } },
    photo_type: { text: "Was it a camera photo, or a screenshot or picture of a document?",
      options: [["Camera photo", { photo_type: "photo" }], ["Screenshot", { photo_type: "screenshot" }], ["Document or bill", { photo_type: "document" }]],
      fn: function (p) { return p.type === "photo"; } },
    has_text: { text: "Was there any writing visible in it?", options: [["Yes, writing", { has_text: true }], ["No writing", { has_text: false }]],
      fn: function (p) { return !!p.ocr_text; } }
  };
  /* Pick the unanswered attribute that splits the remaining candidates most evenly. */
  function nextQuestion(candidates, state, asked) {
    var best = null, bestScore = 0.15;          // skip questions that would barely narrow anything
    Object.keys(QUESTIONS).forEach(function (key) {
      if (asked[key] || state[key] != null) return;
      var vals = candidates.map(QUESTIONS[key].fn).filter(function (v) { return v !== null && v !== undefined; });
      if (vals.length < 6) return;
      var c = count(vals.map(String)), topShare = Math.max.apply(null, Object.keys(c).map(function (x) { return c[x]; })) / vals.length;
      if (1 - topShare > bestScore) { best = key; bestScore = 1 - topShare; }
    });
    return best ? { key: best, text: QUESTIONS[best].text, options: QUESTIONS[best].options } : null;
  }
  /* After repeated misses, drop the most restrictive clue and say which one. */
  function relax(state) {
    var order = ["time", "place", "photo_type", "episode_hint"];
    for (var i = 0; i < order.length; i++) {
      var key = order[i];
      if (has(state[key])) {
        var chip = chips(state).filter(function (c) { return c.key === key; })[0], s = merge(state, {});
        s[key] = EMPTY[key];
        return { state: s, dropped: chip ? chip.value : key };
      }
    }
    return null;
  }
  function episodeTitle(e) { return String((e && e.label) || "").replace(/,\s*\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}$/, "").replace(/\s*\(no camera date\)$/, "").replace(/,\s*[A-Z][a-z]{2}\s+\d{4}$/, ""); }

  return { Library: Library, toks: toks, empty: empty, merge: merge, chips: chips, removeChip: removeChip, clueTypes: clueTypes,
    isVague: isVague, localParse: localParse, nextQuestion: nextQuestion, relax: relax, episodeTitle: episodeTitle,
    monthYear: monthYear, fullDate: fullDate, has: has };
});
