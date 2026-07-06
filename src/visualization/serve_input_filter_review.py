"""Serve an interactive map for reviewing six-hour AIS input filtering.

Run:
    python XGBoost/src/visualization/serve_input_filter_review.py

Open:
    http://127.0.0.1:8796/
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
XGBOOST_DIR = next(
    path for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]
    if path.name == "XGBoost"
)

DEFAULT_DATA_DIR = XGBOOST_DIR / "data" / "pred_data" / "630"
DEFAULT_CENTERLINE = (
    XGBOOST_DIR
    / "src"
    / "model"
    / "downstream_channel_centerline_relaxed_v3_manual_control.geojson"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8796
MAX_CENTERLINE_DISPLAY_POINTS = 1000
MAX_VESSEL_DISPLAY_POINTS = 250


def thin_coordinates(coordinates: list, max_points: int) -> list:
    """Evenly thin display coordinates while retaining both endpoints."""
    if len(coordinates) <= max_points:
        return coordinates
    indexes = {
        round(index * (len(coordinates) - 1) / (max_points - 1))
        for index in range(max_points)
    }
    return [coordinates[index] for index in sorted(indexes)]


def prepare_centerline_for_display(source: dict) -> dict:
    """Keep line features only and reduce browser-side drawing work."""
    if source.get("type") == "FeatureCollection":
        source_features = source.get("features", [])
    elif source.get("type") == "Feature":
        source_features = [source]
    elif source.get("type") in {"LineString", "MultiLineString"}:
        source_features = [{"type": "Feature", "properties": {}, "geometry": source}]
    else:
        source_features = []

    line_features = []
    for feature in source_features:
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        if geometry_type == "LineString":
            display_coordinates = thin_coordinates(
                geometry.get("coordinates", []),
                MAX_CENTERLINE_DISPLAY_POINTS,
            )
        elif geometry_type == "MultiLineString":
            display_coordinates = [
                thin_coordinates(coordinates, MAX_CENTERLINE_DISPLAY_POINTS)
                for coordinates in geometry.get("coordinates", [])
            ]
        else:
            continue

        line_features.append({
            "type": "Feature",
            "properties": feature.get("properties", {}),
            "geometry": {
                "type": geometry_type,
                "coordinates": display_coordinates,
            },
        })

    if not line_features:
        raise ValueError("中心线 GeoJSON 中没有线要素")
    return {"type": "FeatureCollection", "features": line_features}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AIS 输入筛选审查</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body { height:100%; margin:0; font-family:Arial,"Microsoft YaHei",sans-serif; color:#17202a; }
    #app { height:100%; display:grid; grid-template-columns:360px minmax(0,1fr); }
    #side { padding:16px; overflow:auto; border-right:1px solid #ccd3d9; background:#f7f8f9; }
    #map { height:100%; background:#dce6e8; }
    h1 { margin:0 0 5px; font-size:19px; }
    h2 { margin:18px 0 8px; font-size:13px; color:#34495e; }
    .muted { color:#66737f; font-size:12px; line-height:1.55; }
    .controls { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
    button, select { min-height:36px; border:1px solid #aeb8c2; border-radius:5px; background:white; color:#17202a; }
    button { cursor:pointer; padding:0 10px; }
    button.active { color:white; border-color:#176b51; background:#176b51; }
    select { width:100%; padding:0 7px; }
    #vesselSelect { font-variant-numeric:tabular-nums; }
    .summary { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
    .metric { padding:9px; border:1px solid #d7dde2; border-radius:5px; background:white; }
    .metric b { display:block; margin-top:3px; font-size:18px; }
    .row { display:grid; grid-template-columns:1fr auto; gap:8px; padding:6px 0; border-bottom:1px solid #dde2e6; font-size:12px; }
    .row span:last-child { text-align:right; font-weight:600; overflow-wrap:anywhere; }
    .ok { color:#087f5b; } .bad { color:#c0392b; }
    .legend { font-size:12px; line-height:2; }
    .dot { display:inline-block; width:10px; height:10px; margin-right:7px; border-radius:50%; }
    .line { display:inline-block; width:22px; height:4px; margin-right:7px; vertical-align:middle; }
    #message { min-height:19px; margin-top:8px; }
    @media (max-width:760px) {
      #app { grid-template-columns:1fr; grid-template-rows:43% 57%; }
      #side { border-right:0; border-bottom:1px solid #ccd3d9; }
    }
  </style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>AIS 输入筛选审查</h1>
    <div class="muted">最终点用于总览。选择船舶后显示该船六小时 AIS 轨迹及全部采样点。</div>

    <h2>筛选结果</h2>
    <div class="summary">
      <div class="metric">总船舶<b id="totalCount">-</b></div>
      <div class="metric">模型保留<b id="keptCount">-</b></div>
    </div>

    <h2>选择船舶</h2>
    <select id="vesselSelect"><option value="">请选择 MMSI</option></select>
    <button id="clearSelection">清除当前轨迹</button>
    <div id="message" class="muted"></div>

    <h2>筛选明细</h2>
    <div id="detail" class="muted">选择船舶或点击地图上的最终点查看。</div>

    <h2>图例</h2>
    <div class="legend">
      <div><span class="line" style="background:#087fbd"></span>选中船舶六小时轨迹</div>
      <div><span class="line" style="background:#2f3e46"></span>参考中心航线</div>
    </div>
  </aside>
  <main id="map"></main>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map', {preferCanvas:true}).setView([31.6, 120.5], 8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:18,
  attribution:'&copy; OpenStreetMap contributors'
}).addTo(map);

const routeLayer = L.layerGroup().addTo(map);
const centerlineLayer = L.geoJSON(null, {
  style:{color:'#2f3e46', weight:4, opacity:0.85}
}).addTo(map);
let vessels = [];

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({
  '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
}[char]));
const yesNo = value => value ? '<span class="ok">通过</span>' : '<span class="bad">未通过</span>';
const num = (value, digits=2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';

function row(label, value) {
  return `<div class="row"><span>${label}</span><span>${value}</span></div>`;
}

function showDetail(v) {
  document.getElementById('detail').innerHTML =
    row('MMSI', esc(v.mmsi)) +
    row('最终上传时间', esc(v.latest_postime)) +
    row('目的地', esc(v.latest_dest || '-')) +
    row('最终决定', v.keep_for_prediction ? '<span class="ok">保留</span>' : '<span class="bad">剔除</span>') +
    row('筛选原因', esc(v.filter_reason)) +
    row('距中心线', `${num(v.latest_cross_track_km)} km`) +
    row('最终点尚未越过门线', yesNo(v.latest_point_before_gate)) +
    row('目的地匹配上海', yesNo(v.destination_match_shanghai)) +
    row('最终点临近中心线', yesNo(v.final_point_near_centerline)) +
    row('顺流推进达标', yesNo(v.downstream_progress_ok)) +
    row('正向点比例达标', yesNo(v.downstream_ratio_ok)) +
    row('沿线里程变化', `${num(v.route_delta_km)} km`) +
    row('正向移动比例', `${num(Number(v.downstream_positive_ratio) * 100, 1)}%`) +
    row('AIS 点数', esc(v.point_count));
}

async function selectVessel(mmsi) {
  if (!mmsi) {
    routeLayer.clearLayers();
    document.getElementById('detail').textContent = '选择船舶或点击地图上的最终点查看。';
    return;
  }
  document.getElementById('vesselSelect').value = mmsi;
  document.getElementById('message').textContent = `正在加载 ${mmsi}...`;
  const response = await fetch(`/api/vessel?mmsi=${encodeURIComponent(mmsi)}`);
  const payload = await response.json();
  if (!response.ok) {
    document.getElementById('message').textContent = payload.error || '轨迹加载失败';
    return;
  }
  routeLayer.clearLayers();
  const latlngs = payload.points.map(p => [p.lat, p.lon]);
  if (latlngs.length) {
    L.polyline(latlngs, {color:'#087fbd', weight:4, opacity:0.9}).addTo(routeLayer);
    payload.points.forEach((p, index) => {
      const endpoint = index === 0 || index === payload.points.length - 1;
      L.circleMarker([p.lat, p.lon], {
        radius:endpoint ? 6 : 3,
        color:endpoint ? (index === 0 ? '#f39c12' : '#8e44ad') : '#087fbd',
        weight:1, fillOpacity:0.85
      }).bindPopup(`<b>${endpoint ? (index === 0 ? '轨迹起点' : '轨迹终点') : 'AIS 点'}</b><br>
        MMSI: ${esc(mmsi)}<br>时间: ${esc(p.postime)}<br>
        SOG: ${esc(p.sog)} kn<br>COG: ${esc(p.cog)}°<br>
        沿线里程: ${num(p.route_s_km)} km<br>距中心线: ${num(p.cross_track_km)} km`).addTo(routeLayer);
    });
    map.fitBounds(L.latLngBounds(latlngs).pad(0.18), {maxZoom:12});
  }
  showDetail(payload.vessel);
  document.getElementById('message').textContent = `已加载 ${payload.points.length} 个 AIS 点`;
}

async function init() {
  const response = await fetch('/api/overview');
  const data = await response.json();
  vessels = data.vessels;
  centerlineLayer.addData(data.centerline);
  document.getElementById('totalCount').textContent = data.summary.total;
  document.getElementById('keptCount').textContent = data.summary.kept;
  const select = document.getElementById('vesselSelect');
  vessels.slice().sort((a,b) => String(a.mmsi).localeCompare(String(b.mmsi))).forEach(v => {
    const option = document.createElement('option');
    option.value = String(v.mmsi);
    option.textContent = `${v.mmsi} · ${v.keep_for_prediction ? '保留' : '剔除'} · ${v.filter_reason}`;
    select.appendChild(option);
  });
  document.getElementById('message').textContent = '请选择一艘船查看轨迹';
  const bounds = centerlineLayer.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds.pad(0.05));
}

document.getElementById('vesselSelect').onchange = event => selectVessel(event.target.value);
document.getElementById('clearSelection').onclick = () => {
  document.getElementById('vesselSelect').value = '';
  selectVessel('');
  document.getElementById('message').textContent = '已清除当前轨迹';
  const bounds = centerlineLayer.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds.pad(0.05));
};
init().catch(error => { document.getElementById('message').textContent = `加载失败：${error}`; });
</script>
</body>
</html>
"""


def clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to strict JSON records without NaN values."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


class ReviewData:
    def __init__(self, data_dir: Path, centerline_file: Path):
        self.ais_file = data_dir / "ais_6his_6h.csv"
        self.report_file = data_dir / "input_filter_report.csv"
        self.centerline_file = centerline_file

        for path in [self.ais_file, self.report_file, self.centerline_file]:
            if not path.exists():
                raise FileNotFoundError(f"未找到必要文件: {path}")

        self.ais = pd.read_csv(self.ais_file, low_memory=False)
        self.report = pd.read_csv(self.report_file, low_memory=False)
        centerline_source = json.loads(centerline_file.read_text(encoding="utf-8"))
        self.centerline = prepare_centerline_for_display(centerline_source)

        self.ais["mmsi"] = pd.to_numeric(self.ais["mmsi"], errors="coerce").astype("Int64")
        self.report["mmsi"] = pd.to_numeric(self.report["mmsi"], errors="coerce").astype("Int64")
        self.ais = self.ais.dropna(subset=["mmsi", "lon", "lat"]).copy()
        self.report = self.report.dropna(subset=["mmsi", "latest_lon", "latest_lat"]).copy()

        bool_columns = [
            "destination_match_shanghai",
            "final_point_near_centerline",
            "latest_point_before_gate",
            "downstream_progress_ok",
            "downstream_ratio_ok",
            "keep_for_prediction",
        ]
        for column in bool_columns:
            if column in self.report.columns:
                self.report[column] = self.report[column].map(
                    lambda value: str(value).strip().lower() == "true"
                )

        self.ais["postime_sort"] = pd.to_datetime(
            self.ais["postime"], utc=True, errors="coerce"
        )
        self.ais = self.ais.sort_values(["mmsi", "postime_sort"])
        self.report_by_mmsi = self.report.set_index("mmsi", drop=False)

    def overview(self) -> dict:
        columns = [
            "mmsi", "latest_postime", "latest_lon", "latest_lat", "latest_dest",
            "latest_cross_track_km", "point_count", "downstream_positive_ratio",
            "route_delta_km", "destination_match_shanghai",
            "final_point_near_centerline", "latest_point_before_gate",
            "gate_side_cross_value", "downstream_progress_ok",
            "downstream_ratio_ok", "keep_for_prediction", "filter_reason",
        ]
        available = [column for column in columns if column in self.report.columns]
        vessels = clean_records(self.report[available])
        kept = int(self.report["keep_for_prediction"].sum())
        return {
            "summary": {"total": len(self.report), "kept": kept},
            "centerline": self.centerline,
            "vessels": vessels,
        }

    def vessel(self, mmsi_text: str) -> dict:
        try:
            mmsi = int(mmsi_text)
        except ValueError as exc:
            raise KeyError("MMSI 格式无效") from exc

        if mmsi not in self.report_by_mmsi.index:
            raise KeyError(f"筛选报告中没有 MMSI {mmsi}")

        vessel_report = self.report_by_mmsi.loc[mmsi]
        if isinstance(vessel_report, pd.DataFrame):
            vessel_report = vessel_report.iloc[0]

        point_columns = [
            "postime", "lon", "lat", "sog", "cog", "dest",
            "route_s_km", "cross_track_km",
        ]
        available = [column for column in point_columns if column in self.ais.columns]
        points = self.ais.loc[self.ais["mmsi"] == mmsi, available]
        if len(points) > MAX_VESSEL_DISPLAY_POINTS:
            indexes = sorted({
                round(index * (len(points) - 1) / (MAX_VESSEL_DISPLAY_POINTS - 1))
                for index in range(MAX_VESSEL_DISPLAY_POINTS)
            })
            points = points.iloc[indexes]
        return {
            "vessel": clean_records(vessel_report.to_frame().T)[0],
            "points": clean_records(points),
        }


def make_handler(data: ReviewData):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/overview":
                self.send_json(data.overview())
                return
            if parsed.path == "/api/vessel":
                mmsi = parse_qs(parsed.query).get("mmsi", [""])[0]
                try:
                    self.send_json(data.vessel(mmsi))
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=404)
                return
            self.send_json({"error": "Not found"}, status=404)

        def log_message(self, fmt, *args):
            print(f"[map] {self.address_string()} {fmt % args}")

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="启动六小时 AIS 输入筛选审查地图")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main():
    args = parse_args()
    data = ReviewData(args.data_dir.resolve(), args.centerline.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data))
    print(f"AIS 筛选审查地图已启动: http://{args.host}:{args.port}/")
    print(f"数据目录: {args.data_dir.resolve()}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
