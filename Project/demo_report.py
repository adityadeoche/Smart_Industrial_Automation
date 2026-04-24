"""
demo_report.py — Smart Industrial Automation — Evidence Report Generator
========================================================================
Run AFTER run_demo.py is live and the API is responding on port 5000.

Queries all 5 API endpoints, generates a self-contained HTML evidence
report, saves it as demo_report.html, and auto-opens it in the browser.

Purpose
-------
Provides screenshot-quality evidence that the system ran correctly for
inclusion in the research paper appendix.  The single HTML file has no
external dependencies — it can be archived and opened offline.

Usage
-----
  # Terminal 2 (while run_demo.py is running in Terminal 1)
  python demo_report.py

  # Custom API host (e.g. Raspberry Pi on LAN)
  python demo_report.py --host 192.168.1.50 --port 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.error import URLError


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(base_url: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Fetch one API endpoint and return parsed JSON, or None on error."""
    url = f"{base_url}{endpoint}"
    try:
        with urllib_request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except URLError as exc:
        print(f"  [WARN] Could not reach {url}: {exc}")
        return None
    except json.JSONDecodeError as exc:
        print(f"  [WARN] Invalid JSON from {url}: {exc}")
        return None


def fetch_all(base_url: str) -> Dict[str, Any]:
    """Fetch all 5 endpoints and return a dict keyed by endpoint path."""
    endpoints = ["/api/status", "/api/motor/1", "/api/motor/2",
                 "/api/history", "/api/alerts"]
    print(f"[REPORT] Querying API at {base_url} …")
    results = {}
    for ep in endpoints:
        data = _fetch(base_url, ep)
        results[ep] = data
        status = "OK" if data is not None else "FAILED"
        print(f"  {ep:<22} → {status}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HTML report builder — Chart.js powered
# ─────────────────────────────────────────────────────────────────────────────

def _status_badge(status: Optional[str]) -> str:
    colours = {
        "NORMAL"      : ("#00c853", "#e8f5e9"),
        "WARNING"     : ("#ff8f00", "#fff8e1"),
        "CRITICAL"    : ("#d32f2f", "#ffebee"),
        "INITIALISING": ("#607d8b", "#eceff1"),
    }
    fg, bg = colours.get(status or "INITIALISING", ("#607d8b", "#eceff1"))
    return (f'<span style="background:{bg}; color:{fg}; '
            f'border:1px solid {fg}; border-radius:4px; '
            f'padding:2px 10px; font-weight:bold; font-size:13px;">'
            f'{status or "UNKNOWN"}</span>')


def _motor_card(motor_data: Optional[Dict], motor_num: int) -> str:
    if not motor_data:
        return f'<div class="card"><h3>Motor {motor_num}</h3><p>No data</p></div>'

    status = motor_data.get("anomaly_status", "UNKNOWN")
    temp   = motor_data.get("temperature_C")
    vib    = motor_data.get("vibration_x_g")
    rpm    = motor_data.get("speed_rpm")
    flow   = motor_data.get("flow_rate_Lm")
    ia     = motor_data.get("current_a_A")
    ib     = motor_data.get("current_b_A")
    ic     = motor_data.get("current_c_A")
    fault  = motor_data.get("fault_name", "—")
    score  = motor_data.get("anomaly_score")
    rx     = motor_data.get("rx_count", 0)
    mtype  = "Three-Phase Induction Motor" if motor_num == 1 else "Centrifugal Pump Motor"

    speed_row = (f'<tr><td>Speed (RPM)</td><td><b>{rpm:.1f}</b></td></tr>'
                 if rpm is not None else
                 f'<tr><td>Flow Rate (L/min)</td><td><b>{flow:.2f}</b></td></tr>'
                 if flow is not None else "")

    ia_row = (f'<tr><td>Phase-A Current</td><td><b>{ia:.4f} A</b></td></tr>'
              if ia is not None else "")
    ib_row = (f'<tr><td>Phase-B Current</td><td><b>{ib:.4f} A</b></td></tr>'
              if ib is not None else "")
    ic_row = (f'<tr><td>Phase-C Current</td><td><b>{ic:.4f} A</b></td></tr>'
              if ic is not None else "")

    score_str = f"{score:+.6f}" if score is not None else "—"

    return f"""
<div class="card">
  <h3>Motor {motor_num} — {mtype}</h3>
  <p>Status: {_status_badge(status)}</p>
  <table class="info-table">
    <tr><td>Temperature</td><td><b>{f"{temp:.2f} °C" if temp is not None else "—"}</b></td></tr>
    <tr><td>Vibration X</td><td><b>{f"{vib:.4f} g" if vib is not None else "—"}</b></td></tr>
    {speed_row}
    {ia_row}{ib_row}{ic_row}
    <tr><td>Fault Name</td><td><b>{fault}</b></td></tr>
    <tr><td>Anomaly Score</td><td><b>{score_str}</b></td></tr>
    <tr><td>Frames Received</td><td><b>{rx}</b></td></tr>
  </table>
</div>"""


def _alert_row(alert: Dict) -> str:
    status  = alert.get("status", "")
    bg      = "#4a1111" if status == "CRITICAL" else "#3a2f00" if status == "WARNING" else "transparent"
    ts      = alert.get("timestamp", "—")
    motor   = alert.get("motor", "—").replace("motor", "Motor ")
    fault   = alert.get("fault_name", "—")
    temp    = alert.get("temperature_C")
    vib     = alert.get("vibration_x_g")
    score   = alert.get("anomaly_score")
    ia      = alert.get("current_a_A")

    temp_s  = f"{temp:.1f}"   if temp  is not None else "—"
    vib_s   = f"{vib:.4f}"   if vib   is not None else "—"
    score_s = f"{score:+.4f}" if score is not None else "—"
    ia_s    = f"{ia:.2f}"    if ia    is not None else "—"

    return (f'<tr style="background:{bg}">'
            f'<td>{ts}</td><td>{motor}</td>'
            f'<td>{_status_badge(status)}</td>'
            f'<td>{fault}</td><td>{temp_s}</td>'
            f'<td>{vib_s}</td><td>{ia_s}</td><td>{score_s}</td></tr>')


def _collapsible_json(title: str, data: Optional[Dict]) -> str:
    content = json.dumps(data, indent=2, default=str) if data else "No data received."
    return f"""
<details style="margin-top:12px">
  <summary style="cursor:pointer; color:#90caf9; font-weight:bold; padding:6px 0">
    {title}
  </summary>
  <pre class="json-block">{content}</pre>
</details>"""


def _safe(v, decimals=4):
    """Return float rounded to decimals or None — JSON-serialisable."""
    try:
        return round(float(v), decimals) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_html_report(api_data: Dict[str, Any], base_url: str) -> str:
    """Assemble the complete self-contained HTML evidence report with Chart.js charts."""

    generated_at = datetime.now().isoformat(timespec="seconds")

    status_data  = api_data.get("/api/status")  or {}
    motor1_data  = api_data.get("/api/motor/1") or {}
    motor2_data  = api_data.get("/api/motor/2") or {}
    history_data = api_data.get("/api/history") or {}
    alerts_data  = api_data.get("/api/alerts")  or {}

    gateway_start  = status_data.get("gateway_start", "—")
    m1_status_data = status_data.get("motor1") or motor1_data
    m2_status_data = status_data.get("motor2") or motor2_data

    # ── History arrays ────────────────────────────────────────────────────────
    hist_m1 = history_data.get("motor1", [])
    hist_m2 = history_data.get("motor2", [])
    n_m1    = len(hist_m1)
    n_m2    = len(hist_m2)

    def hist_field(hist, key):
        return json.dumps([_safe(r.get(key)) for r in hist])

    labels_m1 = json.dumps(list(range(1, n_m1 + 1)))
    labels_m2 = json.dumps(list(range(1, n_m2 + 1)))

    # Trend data — Motor 1
    m1_temp   = hist_field(hist_m1, "temperature_C")
    m1_vib    = hist_field(hist_m1, "vibration_x_g")
    m1_score  = hist_field(hist_m1, "anomaly_score")
    m1_speed  = hist_field(hist_m1, "speed_rpm")
    m1_ia     = hist_field(hist_m1, "current_a_A")
    m1_ib     = hist_field(hist_m1, "current_b_A")
    m1_ic     = hist_field(hist_m1, "current_c_A")

    # Trend data — Motor 2
    m2_temp   = hist_field(hist_m2, "temperature_C")
    m2_vib    = hist_field(hist_m2, "vibration_x_g")
    m2_score  = hist_field(hist_m2, "anomaly_score")
    m2_flow   = hist_field(hist_m2, "flow_rate_Lm")
    m2_ia     = hist_field(hist_m2, "current_a_A")
    m2_ib     = hist_field(hist_m2, "current_b_A")
    m2_ic     = hist_field(hist_m2, "current_c_A")

    # ── Scatter data (Temp vs Vib, coloured by status) ────────────────────────
    def scatter_pts(hist):
        c_map = {"CRITICAL": "#ff3b3b88", "WARNING": "#ffb30088", "NORMAL": "#00ff8888"}
        pts   = [{"x": _safe(r.get("temperature_C"), 2),
                  "y": _safe(r.get("vibration_x_g"), 4)} for r in hist]
        cols  = [c_map.get(r.get("anomaly_status", "NORMAL"), "#00ff8888") for r in hist]
        return json.dumps(pts), json.dumps(cols)

    sc_pts_m1, sc_cols_m1 = scatter_pts(hist_m1)
    sc_pts_m2, sc_cols_m2 = scatter_pts(hist_m2)

    # ── Fault frequency (bar chart) ───────────────────────────────────────────
    fault_types = ["NORMAL", "BEARING FAULT", "STATOR FAULT", "ROTOR BAR FAULT"]
    fm1 = {f: 0 for f in fault_types}
    fm2 = {f: 0 for f in fault_types}
    alerts = alerts_data.get("alerts", [])
    for a in alerts:
        k = (a.get("fault_name") or "NORMAL").replace("_", " ")
        if a.get("motor") == "motor1" and k in fm1: fm1[k] += 1
        if a.get("motor") == "motor2" and k in fm2: fm2[k] += 1
    bar_m1 = json.dumps([fm1[f] for f in fault_types])
    bar_m2 = json.dumps([fm2[f] for f in fault_types])

    # ── Severity donut ────────────────────────────────────────────────────────
    sev = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
    for a in alerts:
        s = a.get("status", "NORMAL")
        if s in sev: sev[s] += 1
    for h in hist_m1 + hist_m2:
        if h.get("anomaly_status") == "NORMAL": sev["NORMAL"] += 1
    donut_data = json.dumps([sev["NORMAL"], sev["WARNING"], sev["CRITICAL"]])

    # ── Radar (health spider) ─────────────────────────────────────────────────
    def to_radar(motor_data):
        score = _safe(motor_data.get("anomaly_score"), 6) or 0
        temp  = _safe(motor_data.get("temperature_C"), 2) or 65
        vib   = _safe(motor_data.get("vibration_x_g"), 4) or 0.05
        hum   = _safe(motor_data.get("humidity_pct"), 2)  or 50
        spd   = _safe(motor_data.get("speed_rpm") or motor_data.get("flow_rate_Lm"), 2) or 100
        s = round(max(0, min(1, (score + 0.2) / 0.4)) * 100)
        t = round(max(0, min(1, 1 - max(0, temp - 65) / 15)) * 100)
        v = round(max(0, min(1, 1 - max(0, vib - 0.05) / 0.25)) * 100)
        h = round(max(0, min(1, 1 - max(0, hum - 60) / 20)) * 100)
        sp = round(max(0, min(1, spd / 1500)) * 100)
        return json.dumps([s, t, v, h, sp])

    radar_m1 = to_radar(m1_status_data)
    radar_m2 = to_radar(m2_status_data)

    # ── Alert table ───────────────────────────────────────────────────────────
    alert_count = alerts_data.get("count", len(alerts))
    alert_rows  = "".join(_alert_row(a) for a in alerts[:50])
    if not alert_rows:
        alert_rows = '<tr><td colspan="8" style="color:#888">No alerts logged</td></tr>'

    # ── Motor cards ───────────────────────────────────────────────────────────
    card_m1 = _motor_card(m1_status_data, 1)
    card_m2 = _motor_card(m2_status_data, 2)

    # ── Raw JSON collapsibles ─────────────────────────────────────────────────
    raw_sections = "".join([
        _collapsible_json("GET /api/status",  api_data.get("/api/status")),
        _collapsible_json("GET /api/motor/1", api_data.get("/api/motor/1")),
        _collapsible_json("GET /api/motor/2", api_data.get("/api/motor/2")),
        _collapsible_json("GET /api/history", api_data.get("/api/history")),
        _collapsible_json("GET /api/alerts",  api_data.get("/api/alerts")),
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # HTML
    # ─────────────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Industrial Automation — Evidence Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --card-bg: #141922;
    --border:  #1e3050;
    --text:    #c8e8d8;
    --muted:   #5a7a6a;
    --green:   #00ff88;
    --amber:   #ffb300;
    --red:     #ff3b3b;
    --blue:    #00bfff;
    --purple:  #b060ff;
    --orange:  #ff6b35;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background:var(--bg); color:var(--text);
    font-family:'Segoe UI', system-ui, monospace; font-size:13px;
    line-height:1.6; padding:28px 32px;
  }}
  h1 {{ font-size:20px; color:var(--green); margin-bottom:4px;
        text-shadow:0 0 20px rgba(0,255,136,0.4); letter-spacing:0.05em; }}
  h2 {{ font-size:13px; color:var(--muted); margin:28px 0 12px;
        border-bottom:1px solid var(--border); padding-bottom:5px;
        letter-spacing:0.12em; text-transform:uppercase; font-weight:600; }}
  h3 {{ font-size:11px; color:var(--muted); margin-bottom:8px;
        letter-spacing:0.1em; text-transform:uppercase; }}
  .meta {{ color:var(--muted); font-size:11px; margin-bottom:22px; }}
  .cards {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .card {{
    background:var(--card-bg); border:1px solid var(--border);
    border-radius:5px; padding:16px 20px; flex:1; min-width:280px;
  }}
  .info-table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:12px; }}
  .info-table td {{ padding:4px 8px; border-bottom:1px solid #1a2e3d; }}
  .info-table td:first-child {{ color:var(--muted); width:55%; }}
  /* Chart grid */
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:0; }}
  .chart-grid.wide {{ grid-template-columns:1fr; }}
  .chart-panel {{
    background:var(--card-bg); border:1px solid var(--border);
    border-radius:5px; padding:12px;
  }}
  .chart-wrap {{ position:relative; height:220px; }}
  .chart-wrap canvas {{ width:100% !important; height:100% !important; }}
  .chart-wrap.tall {{ height:260px; }}
  /* Trend tabs */
  .trend-tabs {{ display:flex; gap:0; margin-bottom:10px; flex-wrap:wrap; gap:4px; }}
  .trend-tab {{
    background:rgba(0,0,0,0.4); border:1px solid var(--border); color:var(--muted);
    font-size:10px; letter-spacing:0.06em; text-transform:uppercase;
    padding:4px 10px; cursor:pointer; border-radius:2px; transition:all 0.2s;
  }}
  .trend-tab.active {{ background:rgba(0,255,136,0.1); border-color:rgba(0,255,136,0.4); color:var(--green); }}
  /* Alert table */
  .alert-table {{ width:100%; border-collapse:collapse; font-size:11px; margin-top:8px; }}
  .alert-table th {{
    background:var(--border); color:var(--muted); padding:6px 10px;
    text-align:left; font-weight:600; font-size:10px;
    letter-spacing:0.1em; text-transform:uppercase;
  }}
  .alert-table td {{ padding:5px 10px; border-bottom:1px solid #1a2e3d; }}
  .count-badge {{
    display:inline-block; background:rgba(255,179,0,0.1);
    color:var(--amber); border:1px solid rgba(255,179,0,0.3);
    border-radius:12px; padding:2px 12px;
    font-size:11px; font-weight:bold; margin-bottom:10px;
  }}
  .json-block {{
    background:#0a0e14; border:1px solid var(--border); border-radius:4px;
    padding:12px; font-family:'Courier New',monospace; font-size:10px;
    color:#7ec8a0; overflow-x:auto; max-height:360px; overflow-y:auto;
    margin-top:6px; white-space:pre;
  }}
  details summary {{
    cursor:pointer; color:var(--blue); font-weight:bold;
    padding:6px 0; font-size:12px; letter-spacing:0.05em;
  }}
  footer {{
    margin-top:40px; color:var(--muted); font-size:10px;
    border-top:1px solid var(--border); padding-top:10px;
  }}
  @media (max-width:900px) {{
    .chart-grid {{ grid-template-columns:1fr; }}
    .cards {{ flex-direction:column; }}
  }}
</style>
</head>
<body>

<h1>Smart Industrial Automation — Evidence Report</h1>
<div class="meta">
  Generated: {generated_at} &nbsp;|&nbsp;
  Gateway uptime since: {gateway_start} &nbsp;|&nbsp;
  API: {base_url}
</div>

<!-- ═══════════════════════ 1. STATUS ═══════════════════════ -->
<h2>1 — Live System Status (/api/status)</h2>
<div class="cards">
  {card_m1}
  {card_m2}
</div>

<!-- ═══════════════════════ 2. SENSOR TRENDS ══════════════════ -->
<h2>2 — Sensor Trends (/api/history)</h2>

<!-- Motor 1 trends -->
<div class="chart-panel" style="margin-bottom:14px;">
  <h3>Motor 01 — Three-Phase Induction Motor</h3>
  <div class="trend-tabs">
    <button class="trend-tab active" onclick="showTrend('m1','temp',this)">Temperature</button>
    <button class="trend-tab" onclick="showTrend('m1','vib',this)">Vibration X</button>
    <button class="trend-tab" onclick="showTrend('m1','speed',this)">Speed (RPM)</button>
    <button class="trend-tab" onclick="showTrend('m1','score',this)">Anomaly Score</button>
    <button class="trend-tab" onclick="showTrend('m1','current',this)">Current 3-Phase</button>
  </div>
  <div class="chart-wrap"><canvas id="m1-trend-chart"></canvas></div>
  <div id="m1-trend-info" style="font-size:10px;color:var(--muted);margin-top:4px;padding:0 2px;">
    Threshold: 80°C — ISO 60034-1 Class F winding limit
  </div>
</div>

<!-- Motor 2 trends -->
<div class="chart-panel">
  <h3>Motor 02 — Centrifugal Pump Motor</h3>
  <div class="trend-tabs">
    <button class="trend-tab active" onclick="showTrend('m2','temp',this)">Temperature</button>
    <button class="trend-tab" onclick="showTrend('m2','vib',this)">Vibration X</button>
    <button class="trend-tab" onclick="showTrend('m2','flow',this)">Flow Rate</button>
    <button class="trend-tab" onclick="showTrend('m2','score',this)">Anomaly Score</button>
    <button class="trend-tab" onclick="showTrend('m2','current',this)">Current 3-Phase</button>
  </div>
  <div class="chart-wrap"><canvas id="m2-trend-chart"></canvas></div>
  <div id="m2-trend-info" style="font-size:10px;color:var(--muted);margin-top:4px;padding:0 2px;">
    Threshold: 80°C — ISO 60034-1 Class F winding limit
  </div>
</div>

<!-- ═══════════════════════ 3. ANALYSIS ═══════════════════════ -->
<h2>3 — Analysis Charts</h2>
<div class="chart-grid">

  <div class="chart-panel">
    <h3>Health Radar — Multi-Sensor Snapshot</h3>
    <div class="chart-wrap"><canvas id="chart-radar"></canvas></div>
  </div>

  <div class="chart-panel">
    <h3>Scatter — Temperature vs Vibration X</h3>
    <div class="chart-wrap"><canvas id="chart-scatter"></canvas></div>
  </div>

  <div class="chart-panel">
    <h3>Fault Frequency — Alert History</h3>
    <div class="chart-wrap"><canvas id="chart-bar"></canvas></div>
  </div>

  <div class="chart-panel">
    <h3>Alert Severity Distribution</h3>
    <div class="chart-wrap"><canvas id="chart-donut"></canvas></div>
  </div>

</div>

<!-- ═══════════════════════ 4. ALERTS ════════════════════════ -->
<h2>4 — Alert Event Log (/api/alerts)</h2>
<div class="count-badge">{alert_count} anomaly events logged</div>
<table class="alert-table">
  <thead>
    <tr>
      <th>Timestamp</th><th>Motor</th><th>Status</th><th>Fault</th>
      <th>Temp (°C)</th><th>Vib (g)</th><th>Ia (A)</th><th>Score</th>
    </tr>
  </thead>
  <tbody>
    {alert_rows}
  </tbody>
</table>

<!-- ═══════════════════════ 5. RAW JSON ══════════════════════ -->
<h2>5 — Raw API Responses (click to expand)</h2>
<p style="color:var(--muted);font-size:11px;margin-bottom:4px">
  Exact JSON payloads as returned by the gateway — for examiner verification of the API contract.
</p>
{raw_sections}

<footer>
  Smart Industrial Automation Project — Evidence Report &nbsp;|&nbsp;
  Generated by demo_report.py on {generated_at}
</footer>

<!-- ═══════════════════════ CHART.JS ════════════════════════ -->
<script>
'use strict';

/* ── Shared data injected from Python ─────────────────────── */
const LABELS_M1  = {labels_m1};
const LABELS_M2  = {labels_m2};

const DATA = {{
  m1: {{
    temp   : {m1_temp},
    vib    : {m1_vib},
    score  : {m1_score},
    speed  : {m1_speed},
    ia     : {m1_ia},
    ib     : {m1_ib},
    ic     : {m1_ic},
  }},
  m2: {{
    temp   : {m2_temp},
    vib    : {m2_vib},
    score  : {m2_score},
    flow   : {m2_flow},
    ia     : {m2_ia},
    ib     : {m2_ib},
    ic     : {m2_ic},
  }},
  scatter_m1_pts  : {sc_pts_m1},
  scatter_m1_cols : {sc_cols_m1},
  scatter_m2_pts  : {sc_pts_m2},
  scatter_m2_cols : {sc_cols_m2},
  bar_m1   : {bar_m1},
  bar_m2   : {bar_m2},
  donut    : {donut_data},
  radar_m1 : {radar_m1},
  radar_m2 : {radar_m2},
}};

/* ── Shared Chart.js options ──────────────────────────────── */
const FONT = {{ family:'monospace', size:9 }};
const TOOLTIP = {{
  backgroundColor:'#0a0e14', borderColor:'#1e3050', borderWidth:1,
  titleColor:'#00ff88', bodyColor:'#c8e8d8', titleFont:{{ family:'monospace', size:10 }},
}};
const GRID_COLOR = '#0d1e29';
const TICK_COLOR = '#3d6050';

function makeScales(xLabel, yLabel) {{
  return {{
    x: {{ title:{{display:!!xLabel, text:xLabel||'', color:TICK_COLOR, font:FONT}},
           ticks:{{color:TICK_COLOR, font:FONT, maxTicksLimit:12}}, grid:{{color:GRID_COLOR}} }},
    y: {{ title:{{display:!!yLabel, text:yLabel||'', color:TICK_COLOR, font:FONT}},
           ticks:{{color:TICK_COLOR, font:FONT}}, grid:{{color:GRID_COLOR}} }},
  }};
}}

/* ── Trend chart state ────────────────────────────────────── */
const _trendCharts = {{ m1: null, m2: null }};
const _trendState  = {{ m1: 'temp', m2: 'temp' }};

const TREND_CFG = {{
  temp  : {{ label:'Temperature (°C)',  color:'#ff6b35', thresh:80,    threshLabel:'80°C — ISO 60034-1' }},
  vib   : {{ label:'Vibration X (g)',   color:'#00bfff', thresh:0.3,   threshLabel:'0.3g — ISO 10816-3 Zone D' }},
  score : {{ label:'Anomaly Score',     color:'#ffb300', thresh:-0.10, threshLabel:'-0.10 — Isolation Forest CRITICAL boundary' }},
  speed : {{ label:'Speed (RPM)',       color:'#00ff88', thresh:null,  threshLabel:'No fixed threshold' }},
  flow  : {{ label:'Flow Rate (L/min)', color:'#00ff88', thresh:null,  threshLabel:'No fixed threshold' }},
  current:{{ label:'Current (A)',       color:null,      thresh:null,  threshLabel:'Phase imbalance > 5% PCUR indicates stator fault (IEC 60034-26)' }},
}};

function showTrend(motor, ch, btn) {{
  // Update active tab
  const panel = btn.closest('.chart-panel');
  panel.querySelectorAll('.trend-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  _trendState[motor] = ch;
  _buildTrendChart(motor, ch);
}}

function _buildTrendChart(motor, ch) {{
  const labels  = motor === 'm1' ? LABELS_M1 : LABELS_M2;
  const d       = DATA[motor];
  const cfg     = TREND_CFG[ch] || TREND_CFG.temp;
  const infoEl  = document.getElementById(`${{motor}}-trend-info`);
  if (infoEl) infoEl.textContent = 'Threshold: ' + cfg.threshLabel;

  let datasets;

  if (ch === 'current') {{
    /* Three-phase overlay */
    datasets = [
      {{ label:'Phase A (Ia)', data:d.ia, borderColor:'#ff6b35', backgroundColor:'rgba(255,107,53,0.06)',
         borderWidth:1.5, pointRadius:0, tension:0.2, fill:false }},
      {{ label:'Phase B (Ib)', data:d.ib, borderColor:'#00ff88', backgroundColor:'rgba(0,255,136,0.06)',
         borderWidth:1.5, pointRadius:0, tension:0.2, fill:false }},
      {{ label:'Phase C (Ic)', data:d.ic, borderColor:'#00bfff', backgroundColor:'rgba(0,191,255,0.06)',
         borderWidth:1.5, pointRadius:0, tension:0.2, fill:false }},
    ];
  }} else {{
    const values = d[ch] || [];
    datasets = [{{
      label: cfg.label, data: values,
      borderColor: cfg.color, backgroundColor: cfg.color + '1a',
      borderWidth:1.5, pointRadius:0, tension:0.3, fill:true,
    }}];
    if (cfg.thresh !== null) {{
      datasets.push({{
        label:'Threshold', data: labels.map(() => cfg.thresh),
        borderColor:'#ff3b3b', borderWidth:1, borderDash:[4,4],
        pointRadius:0, fill:false,
      }});
    }}
  }}

  const canvasId = `${{motor}}-trend-chart`;
  const existing = _trendCharts[motor];
  if (existing) {{ existing.destroy(); }}

  _trendCharts[motor] = new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{{ legend:{{ labels:{{ color:TICK_COLOR, font:FONT, boxWidth:10 }} }}, tooltip:TOOLTIP }},
      scales: makeScales('Sample', cfg.label),
    }},
  }});
}}

/* ── Radar ────────────────────────────────────────────────── */
new Chart(document.getElementById('chart-radar'), {{
  type: 'radar',
  data: {{
    labels: ['Score','Temp','Vibration','Humidity','Speed/Flow'],
    datasets: [
      {{ label:'Motor 01', data:DATA.radar_m1, borderColor:'#00ff88',
         backgroundColor:'rgba(0,255,136,0.07)', pointBackgroundColor:'#00ff88', borderWidth:1.5 }},
      {{ label:'Motor 02', data:DATA.radar_m2, borderColor:'#00bfff',
         backgroundColor:'rgba(0,191,255,0.07)', pointBackgroundColor:'#00bfff', borderWidth:1.5 }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{ legend:{{ labels:{{ color:TICK_COLOR, font:FONT, boxWidth:10 }} }}, tooltip:TOOLTIP }},
    scales:{{ r:{{
      min:0, max:100,
      ticks:{{ color:TICK_COLOR, font:FONT, stepSize:25, backdropColor:'transparent' }},
      grid:{{ color:'#1a2e3d' }}, pointLabels:{{ color:TICK_COLOR, font:FONT }},
      angleLines:{{ color:'#1a2e3d' }}
    }} }}
  }}
}});

/* ── Scatter ──────────────────────────────────────────────── */
new Chart(document.getElementById('chart-scatter'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{ label:'Motor 01', data:DATA.scatter_m1_pts, backgroundColor:DATA.scatter_m1_cols,
         pointRadius:4, borderWidth:0 }},
      {{ label:'Motor 02', data:DATA.scatter_m2_pts, backgroundColor:DATA.scatter_m2_cols,
         pointRadius:3, borderWidth:0, pointStyle:'triangle' }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{ legend:{{ labels:{{ color:TICK_COLOR, font:FONT, boxWidth:10 }} }}, tooltip:TOOLTIP }},
    scales: makeScales('Temperature (°C)', 'Vibration X (g)'),
  }}
}});

/* ── Bar ──────────────────────────────────────────────────── */
new Chart(document.getElementById('chart-bar'), {{
  type: 'bar',
  data: {{
    labels: ['NORMAL','BEARING\\nFAULT','STATOR\\nFAULT','ROTOR BAR\\nFAULT'],
    datasets: [
      {{ label:'Motor 01', data:DATA.bar_m1, backgroundColor:'rgba(0,255,136,0.45)', borderColor:'#00ff88', borderWidth:1 }},
      {{ label:'Motor 02', data:DATA.bar_m2, backgroundColor:'rgba(0,191,255,0.45)', borderColor:'#00bfff', borderWidth:1 }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{ legend:{{ labels:{{ color:TICK_COLOR, font:FONT, boxWidth:10 }} }}, tooltip:TOOLTIP }},
    scales:{{
      x:{{ ticks:{{ color:TICK_COLOR, font:FONT }}, grid:{{ color:GRID_COLOR }} }},
      y:{{ beginAtZero:true, ticks:{{ color:TICK_COLOR, font:FONT, stepSize:1 }}, grid:{{ color:GRID_COLOR }} }},
    }}
  }}
}});

/* ── Donut ────────────────────────────────────────────────── */
new Chart(document.getElementById('chart-donut'), {{
  type: 'doughnut',
  data: {{
    labels: ['NORMAL','WARNING','CRITICAL'],
    datasets: [{{
      data: DATA.donut,
      backgroundColor: ['rgba(0,194,102,0.55)','rgba(255,179,0,0.55)','rgba(255,59,59,0.55)'],
      borderColor: ['#00c266','#ffb300','#ff3b3b'], borderWidth:1,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    cutout:'65%',
    plugins:{{
      legend:{{ position:'bottom', labels:{{ color:TICK_COLOR, font:FONT, padding:12, boxWidth:10 }} }},
      tooltip:TOOLTIP,
    }}
  }}
}});

/* ── Build initial trend charts ──────────────────────────── */
_buildTrendChart('m1', 'temp');
_buildTrendChart('m2', 'temp');

</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Industrial Automation — Evidence Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost",
                        help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000,
                        help="API port (default: 5000)")
    parser.add_argument("--output", default="demo_report.html",
                        help="Output HTML file path")
    parser.add_argument("--no-browser", action="store_true",
                        help="Skip auto-opening the report in a browser")
    return parser.parse_args()


def main() -> None:
    args     = _parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print()
    print("═" * 60)
    print("  Smart Industrial Automation — Evidence Report Generator")
    print("═" * 60)
    print()

    # Fetch all API endpoints
    api_data = fetch_all(base_url)

    # Check at least /api/status responded
    if api_data.get("/api/status") is None:
        print()
        print("[ERROR] Could not reach /api/status.")
        print(f"        Make sure run_demo.py is running on {base_url}")
        sys.exit(1)

    # Build report
    print()
    print("[REPORT] Building HTML report …")
    html = build_html_report(api_data, base_url)

    # Save
    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[REPORT] Saved: {output_path}")

    # Open in browser
    if not args.no_browser:
        webbrowser.open("file://" + output_path)
        print(f"[REPORT] Opened in browser.")

    print()
    print("[REPORT] Done.  File can be archived and opened offline.")
    print()


if __name__ == "__main__":
    main()