// Receives anonymous usage events (ids, clue types and counts only; never what the person typed) and writes
// them to the function log, where they can be exported to compute the funnel metrics.
module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).end();
  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const events = Array.isArray(body.events) ? body.events.slice(0, 20) : [];
    for (const e of events) console.log("EVENT " + JSON.stringify(e).slice(0, 1500));
  } catch (e) { /* logging must never break the flow */ }
  res.status(204).end();
};
