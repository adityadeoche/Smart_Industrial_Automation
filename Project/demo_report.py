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
# SVG sparkline builder
# ─────────────────────────────────────────────────────────────────────────────

def _sparkline_svg(readings: List[Dict], width: int = 400, height: int = 120,
                   motor_label: str = "Motor") -> str:
    """
    Build a pure-SVG inline sparkline from history readings.
    Plots temperature_C (blue line) and anomaly_score (red line).
    No external chart library — pure SVG path elements.
    """
    if not readings:
        return (f'<svg width="{width}" height="{height}" '
                f'xmlns="http://www.w3.org/2000/svg">'
                f'<text x="10" y="60" fill="#888" font-size="12">'
                f'No history data</text></svg>')

    temps  = [r.get("temperature_C", 0) or 0 for r in readings]
    scores = [r.get("anomaly_score", 0) or 0 for r in readings]
    n      = len(readings)

    pad    = 30   # left/bottom padding for axis labels
    w      = width  - pad - 10
    h      = height - pad - 10

    def normalise(vals: List[float]) -> List[float]:
        lo, hi = min(vals), max(vals)
        span   = hi - lo or 1.0
        return [(v - lo) / span for v in vals]

    def build_path(norm_vals: List[float]) -> str:
        pts = []
        for i, v in enumerate(norm_vals):
            x = pad + (i / max(n - 1, 1)) * w
            y = h + 10 - v * h          # SVG y-axis is inverted
            pts.append(f"{x:.1f},{y:.1f}")
        return "M " + " L ".join(pts)

    t_norm = normalise(temps)
    s_norm = normalise(scores)

    t_path = build_path(t_norm)
    s_path = build_path(s_norm)

    t_min, t_max = min(temps),  max(temps)
    s_min, s_max = min(scores), max(scores)

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
     style="background:#12122a; border-radius:6px; overflow:visible">

  <!-- Axis lines -->
  <line x1="{pad}" y1="10" x2="{pad}" y2="{h+10}" stroke="#333" stroke-width="1"/>
  <line x1="{pad}" y1="{h+10}" x2="{pad+w}" y2="{h+10}" stroke="#333" stroke-width="1"/>

  <!-- Temperature sparkline (blue) -->
  <path d="{t_path}" fill="none" stroke="#4fc3f7" stroke-width="1.8"
        stroke-linejoin="round" stroke-linecap="round"/>

  <!-- Score sparkline (red) -->
  <path d="{s_path}" fill="none" stroke="#ef5350" stroke-width="1.8"
        stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="3,2"/>

  <!-- Legend -->
  <rect x="{pad+4}" y="14" width="10" height="3" fill="#4fc3f7" rx="1"/>
  <text x="{pad+17}" y="19" fill="#4fc3f7" font-size="10" font-family="monospace">
    Temp ({t_min:.1f}–{t_max:.1f} °C)
  </text>
  <rect x="{pad+4}" y="26" width="10" height="3" fill="#ef5350" rx="1"/>
  <text x="{pad+17}" y="31" fill="#ef5350" font-size="10" font-family="monospace">
    Score ({s_min:.4f}–{s_max:.4f})
  </text>

  <!-- Y-axis labels -->
  <text x="{pad-4}" y="15" fill="#666" font-size="8" text-anchor="end">hi</text>
  <text x="{pad-4}" y="{h+12}" fill="#666" font-size="8" text-anchor="end">lo</text>

  <!-- Motor label -->
  <text x="{pad+w}" y="{h+24}" fill="#888" font-size="9"
        text-anchor="end" font-family="monospace">{motor_label} — last {n} readings</text>
</svg>"""
    return svg


# ─────────────────────────────────────────────────────────────────────────────
# HTML report builder
# ─────────────────────────────────────────────────────────────────────────────

def _status_badge(status: Optional[str]) -> str:
    colours = {
        "NORMAL"     : ("#00c853", "#e8f5e9"),
        "WARNING"    : ("#ff8f00", "#fff8e1"),
        "CRITICAL"   : ("#d32f2f", "#ffebee"),
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

    status     = motor_data.get("anomaly_status", "UNKNOWN")
    temp       = motor_data.get("temperature_C")
    vib        = motor_data.get("vibration_x_g")
    rpm        = motor_data.get("speed_rpm")
    flow       = motor_data.get("flow_rate_Lm")
    fault      = motor_data.get("fault_name", "—")
    score      = motor_data.get("anomaly_score")
    rx         = motor_data.get("rx_count", 0)
    mtype      = "Three-Phase Induction Motor" if motor_num == 1 else "Centrifugal Pump Motor"

    speed_row = (f'<tr><td>Speed (RPM)</td><td><b>{rpm:.1f}</b></td></tr>'
                 if rpm is not None else
                 f'<tr><td>Flow Rate (L/min)</td><td><b>{flow:.2f}</b></td></tr>'
                 if flow is not None else "")

    score_str = f"{score:+.6f}" if score is not None else "—"

    return f"""
<div class="card">
  <h3>Motor {motor_num} — {mtype}</h3>
  <p>Status: {_status_badge(status)}</p>
  <table class="info-table">
    <tr><td>Temperature</td><td><b>{f"{temp:.2f} °C" if temp is not None else "—"}</b></td></tr>
    <tr><td>Vibration X</td><td><b>{f"{vib:.4f} g" if vib is not None else "—"}</b></td></tr>
    {speed_row}
    <tr><td>Fault Name</td><td><b>{fault}</b></td></tr>
    <tr><td>Anomaly Score</td><td><b>{score_str}</b></td></tr>
    <tr><td>Frames Received</td><td><b>{rx}</b></td></tr>
  </table>
</div>"""


def _alert_row(alert: Dict) -> str:
    status = alert.get("status", "")
    bg     = "#4a1111" if status == "CRITICAL" else "#3a2f00" if status == "WARNING" else "transparent"
    ts     = alert.get("timestamp", "—")
    motor  = alert.get("motor", "—").replace("motor", "Motor ")
    fault  = alert.get("fault_name", "—")
    temp   = alert.get("temperature_C")
    vib    = alert.get("vibration_x_g")
    score  = alert.get("anomaly_score")

    temp_s  = f"{temp:.1f}" if temp  is not None else "—"
    vib_s   = f"{vib:.4f}"  if vib   is not None else "—"
    score_s = f"{score:+.4f}" if score is not None else "—"

    return (f'<tr style="background:{bg}">'
            f'<td>{ts}</td><td>{motor}</td>'
            f'<td>{_status_badge(status)}</td>'
            f'<td>{fault}</td><td>{temp_s}</td>'
            f'<td>{vib_s}</td><td>{score_s}</td></tr>')


def _collapsible_json(title: str, data: Optional[Dict]) -> str:
    content = json.dumps(data, indent=2, default=str) if data else "No data received."
    return f"""
<details style="margin-top:12px">
  <summary style="cursor:pointer; color:#90caf9; font-weight:bold; padding:6px 0">
    {title}
  </summary>
  <pre class="json-block">{content}</pre>
</details>"""


def build_html_report(api_data: Dict[str, Any], base_url: str) -> str:
    """Assemble the complete self-contained HTML evidence report."""

    generated_at = datetime.now().isoformat(timespec="seconds")

    status_data  = api_data.get("/api/status") or {}
    motor1_data  = (api_data.get("/api/motor/1") or {})
    motor2_data  = (api_data.get("/api/motor/2") or {})
    history_data = api_data.get("/api/history") or {}
    alerts_data  = api_data.get("/api/alerts") or {}

    gateway_start = status_data.get("gateway_start", "—")
    m1_status_data = status_data.get("motor1") or motor1_data
    m2_status_data = status_data.get("motor2") or motor2_data

    # History sparklines
    hist_m1  = history_data.get("motor1", [])
    hist_m2  = history_data.get("motor2", [])
    svg_m1   = _sparkline_svg(hist_m1, motor_label="Motor 1")
    svg_m2   = _sparkline_svg(hist_m2, motor_label="Motor 2")

    # Alert table
    alerts     = alerts_data.get("alerts", [])
    alert_count = alerts_data.get("count", len(alerts))
    alert_rows  = "".join(_alert_row(a) for a in alerts[:50])   # cap at 50 rows
    if not alert_rows:
        alert_rows = '<tr><td colspan="7" style="color:#888">No alerts logged</td></tr>'

    # Motor cards
    card_m1 = _motor_card(m1_status_data, 1)
    card_m2 = _motor_card(m2_status_data, 2)

    # Raw JSON collapsibles
    raw_sections = "".join([
        _collapsible_json("GET /api/status",   api_data.get("/api/status")),
        _collapsible_json("GET /api/motor/1",  api_data.get("/api/motor/1")),
        _collapsible_json("GET /api/motor/2",  api_data.get("/api/motor/2")),
        _collapsible_json("GET /api/history",  api_data.get("/api/history")),
        _collapsible_json("GET /api/alerts",   api_data.get("/api/alerts")),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Industrial Automation — Live System Evidence Report</title>
<style>
  :root {{
    --bg:      #1a1a2e;
    --card-bg: #16213e;
    --accent:  #0f3460;
    --border:  #0f3460;
    --text:    #e0e0e0;
    --muted:   #888;
    --blue:    #4fc3f7;
    --green:   #00c853;
    --amber:   #ff8f00;
    --red:     #ef5350;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    padding: 32px;
  }}
  h1 {{ font-size: 22px; color: var(--blue); margin-bottom: 4px; }}
  h2 {{ font-size: 16px; color: var(--blue); margin: 28px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ font-size: 14px; color: #b0c4de; margin-bottom: 10px; }}
  .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 22px;
    flex: 1; min-width: 280px;
  }}
  .info-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
  .info-table td {{ padding: 4px 8px; border-bottom: 1px solid #1f2d4a; }}
  .info-table td:first-child {{ color: var(--muted); width: 55%; }}
  .alert-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  .alert-table th {{
    background: var(--accent); color: var(--blue);
    padding: 7px 10px; text-align: left; font-weight: 600;
  }}
  .alert-table td {{ padding: 5px 10px; border-bottom: 1px solid #1f2d4a; }}
  .sparklines {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 8px; }}
  .sparkline-box {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
  .json-block {{
    background: #0d1117; border: 1px solid var(--border);
    border-radius: 6px; padding: 14px;
    font-family: 'Courier New', monospace; font-size: 11px;
    color: #a8d8a8; overflow-x: auto;
    max-height: 400px; overflow-y: auto;
    margin-top: 6px;
    white-space: pre;
  }}
  .count-badge {{
    display: inline-block;
    background: var(--accent); color: var(--blue);
    border-radius: 12px; padding: 2px 12px;
    font-size: 12px; font-weight: bold;
    margin-bottom: 10px;
  }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--border); padding-top: 12px; }}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- Section 1: Header                                       -->
<!-- ═══════════════════════════════════════════════════════ -->
<h1>Smart Industrial Automation — Live System Evidence Report</h1>
<div class="meta">
  Generated: {generated_at} &nbsp;|&nbsp;
  Gateway uptime since: {gateway_start} &nbsp;|&nbsp;
  API base: {base_url}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- Section 2: System Status Panel                          -->
<!-- ═══════════════════════════════════════════════════════ -->
<h2>System Status (from /api/status)</h2>
<div class="cards">
  {card_m1}
  {card_m2}
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- Section 3: Alert Log                                    -->
<!-- ═══════════════════════════════════════════════════════ -->
<h2>Alert Log (from /api/alerts)</h2>
<div class="count-badge">{alert_count} anomaly events logged</div>
<table class="alert-table">
  <thead>
    <tr>
      <th>Timestamp</th>
      <th>Motor</th>
      <th>Status</th>
      <th>Fault</th>
      <th>Temp (°C)</th>
      <th>Vib (g)</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    {alert_rows}
  </tbody>
</table>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- Section 4: History Charts                               -->
<!-- ═══════════════════════════════════════════════════════ -->
<h2>History Charts (from /api/history)</h2>
<p style="color:var(--muted); font-size:12px; margin-bottom:8px">
  Blue = Temperature (°C) &nbsp;|&nbsp; Red dashed = Anomaly Score
</p>
<div class="sparklines">
  <div class="sparkline-box">
    <h3>Motor 1 — Three-Phase Induction Motor</h3>
    {svg_m1}
  </div>
  <div class="sparkline-box">
    <h3>Motor 2 — Centrifugal Pump Motor</h3>
    {svg_m2}
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- Section 5: Raw API Responses                            -->
<!-- ═══════════════════════════════════════════════════════ -->
<h2>Raw API Responses (click to expand)</h2>
<p style="color:var(--muted); font-size:12px; margin-bottom:4px">
  Exact JSON payloads as returned by the gateway — for examiner verification
  of the API contract.
</p>
{raw_sections}

<footer>
  Smart Industrial Automation Project — Evidence Report<br>
  Generated by demo_report.py on {generated_at}
</footer>

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