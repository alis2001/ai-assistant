import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import httpx


LOG_PATH = os.environ.get("CF_RESULTS_LOG", "/home/ai/whisper/cf_results.jsonl")
RESULTS_URL = os.environ.get("RESULTS_URL")  # Optional: if set, fetch from remote URL
INGEST_TOKEN = os.environ.get("INGEST_TOKEN")  # Optional shared secret for write access

app = FastAPI(title="CF Results")


def read_results(max_items: int = 200):
    results = []
    try:
        if RESULTS_URL:
            # Fetch newline-delimited JSON from remote URL
            with httpx.Client(timeout=3.0) as client:
                r = client.get(RESULTS_URL, headers={"Cache-Control": "no-cache"})
                r.raise_for_status()
                for line in r.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        results.append(obj)
                    except Exception:
                        continue
        else:
            if not os.path.exists(LOG_PATH):
                return results
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        results.append(obj)
                    except Exception:
                        continue
        # Sort newest first by timestamp
        def parse_ts(s: str):
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.min
        results.sort(key=lambda r: parse_ts(r.get("timestamp", "")), reverse=True)
        return results[:max_items]
    except Exception:
        return []


def it_datetime(dt: datetime) -> str:
    # Italian format: DD/MM/YYYY HH:MM:SS
    return dt.strftime("%d/%m/%Y %H:%M:%S")


def render_table(rows):
    # Minimal white styling
    style = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #fff; color: #111; margin: 0; padding: 24px; }
      h1 { font-size: 20px; margin: 0 0 16px; }
      .container { max-width: 880px; margin: 0 auto; }
      table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }
      th, td { padding: 10px 12px; border-bottom: 1px solid #f1f5f9; text-align: left; font-size: 14px; }
      th { background: #f8fafc; color: #111827; position: sticky; top: 0; }
      tr:hover td { background: #f9fafb; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
      .ok { background: #e6fce6; color: #14532d; border: 1px solid #bbf7d0; }
      .warn { background: #fff1f2; color: #7f1d1d; border: 1px solid #fecdd3; }
      .muted { color: #6b7280; font-size: 12px; }
    </style>
    """
    head = """
      <div class="container">
        <h1>Codici Fiscali elaborati</h1>
        <table>
          <thead>
            <tr>
              <th>Data e ora</th>
              <th>Codice Fiscale</th>
              <th>Completezza</th>
              <th>Confidenza</th>
            </tr>
          </thead>
          <tbody id="rows">
    """
    body_rows = []
    for r in rows:
        ts_raw = r.get("timestamp", "")
        try:
            dt = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
            ts_it = it_datetime(dt)
        except Exception:
            ts_it = ts_raw

        cf = r.get("cf_code", "").strip()
        is_complete = bool(r.get("is_complete", False))
        conf = r.get("confidence", 0.0)
        conf_str = f"{conf:.2f}"
        badge = f"<span class='badge ok'>Completo</span>" if is_complete else f"<span class='badge warn'>Incompleto</span>"

        body_rows.append(
            f"<tr><td>{ts_it}</td><td><strong>{cf}</strong></td><td>{badge}</td><td>{conf_str}</td></tr>"
        )

    tail = """
          </tbody>
        </table>
      </div>
      <script>
        async function refreshTable() {
          try {
            const res = await fetch('/api/results', { cache: 'no-store' });
            const data = await res.json();
            const tbody = document.getElementById('rows');
            if (!Array.isArray(data)) return;
            tbody.innerHTML = data.map(r => {
              const badge = r.is_complete ? "<span class='badge ok'>Completo</span>" : "<span class='badge warn'>Incompleto</span>";
              const conf = (typeof r.confidence === 'number') ? r.confidence.toFixed(2) : '';
              const cf = (r.cf_code || '').trim();
              const ts = r.timestamp_it || r.timestamp || '';
              return `<tr><td>${ts}</td><td><strong>${cf}</strong></td><td>${badge}</td><td>${conf}</td></tr>`;
            }).join('');
          } catch (e) {
            // Fail silently
          }
        }
        setInterval(refreshTable, 3000);
      </script>
    """
    return style + head + "\n".join(body_rows) + tail


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    rows = read_results()
    html = render_table(rows)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@app.get("/api/results")
async def api_results():
    rows = read_results()
    payload = []
    for r in rows:
        ts_raw = r.get("timestamp", "")
        try:
            dt = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
            ts_it = it_datetime(dt)
        except Exception:
            ts_it = ts_raw
        payload.append({
            "timestamp": ts_raw,
            "timestamp_it": ts_it,
            "cf_code": r.get("cf_code", ""),
            "is_complete": bool(r.get("is_complete", False)),
            "confidence": float(r.get("confidence", 0.0)),
            "length": int(r.get("length", 0)),
        })
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.post("/api/ingest")
async def api_ingest(request: Request, authorization: str | None = Header(default=None)):
    # Simple bearer token check if provided
    if INGEST_TOKEN:
        expected = f"Bearer {INGEST_TOKEN}"
        if authorization != expected:
            return PlainTextResponse("Unauthorized", status_code=401)
    try:
        body = await request.json()
        # Accept either a single object or a list of objects
        items = body if isinstance(body, list) else [body]
        # Minimal normalization
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                # Ensure required fields exist
                obj.setdefault("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
                obj.setdefault("cf_code", "")
                obj.setdefault("is_complete", False)
                obj.setdefault("confidence", 0.0)
                obj.setdefault("length", len(obj.get("cf_code", "")))
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return PlainTextResponse("OK")
    except Exception as e:
        return PlainTextResponse(f"Bad Request: {e}", status_code=400)


