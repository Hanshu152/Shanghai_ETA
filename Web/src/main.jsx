import React from 'react';
import { createRoot } from 'react-dom/client';
import { MapContainer, Marker, Polyline, Popup, TileLayer } from 'react-leaflet';
import ReactECharts from 'echarts-for-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './styles.css';

const voyages = [
  {
    id: 'COSCO-SEA-716E',
    vessel: 'COSCO Shipping Aries',
    service: 'AEU3',
    origin: 'Singapore',
    destination: 'Shanghai Yangshan',
    status: 'In transit',
    eta: '07-08 18:20',
    predictedEta: '07-08 17:45',
    delayRisk: 'Low',
    confidence: 91,
    error: -35,
    speed: 17.8,
    progress: 78,
    position: [27.4, 123.2],
    route: [
      [1.26, 103.84],
      [9.9, 109.1],
      [18.4, 116.9],
      [24.2, 121.7],
      [30.62, 122.06]
    ]
  },
  {
    id: 'COSCO-NAV-221W',
    vessel: 'COSCO Netherlands',
    service: 'TP9',
    origin: 'Busan',
    destination: 'Ningbo Zhoushan',
    status: 'Berthing queue',
    eta: '07-08 06:10',
    predictedEta: '07-08 07:25',
    delayRisk: 'Medium',
    confidence: 83,
    error: 75,
    speed: 11.6,
    progress: 92,
    position: [29.6, 122.9],
    route: [
      [35.1, 129.05],
      [32.8, 126.6],
      [30.7, 124.4],
      [29.9, 122.8],
      [29.87, 122.16]
    ]
  },
  {
    id: 'COSCO-PAC-408N',
    vessel: 'COSCO Pacific',
    service: 'CEN',
    origin: 'Kaohsiung',
    destination: 'Qingdao',
    status: 'Weather watch',
    eta: '07-09 11:30',
    predictedEta: '07-09 14:05',
    delayRisk: 'High',
    confidence: 76,
    error: 155,
    speed: 14.2,
    progress: 55,
    position: [25.9, 121.9],
    route: [
      [22.62, 120.28],
      [24.1, 121.2],
      [27.8, 122.7],
      [32.1, 123.4],
      [36.06, 120.38]
    ]
  },
  {
    id: 'COSCO-IND-033E',
    vessel: 'COSCO Indonesia',
    service: 'AEX7',
    origin: 'Hong Kong',
    destination: 'Xiamen',
    status: 'On schedule',
    eta: '07-07 22:40',
    predictedEta: '07-07 22:15',
    delayRisk: 'Low',
    confidence: 94,
    error: -25,
    speed: 16.4,
    progress: 84,
    position: [23.7, 118.4],
    route: [
      [22.31, 114.16],
      [22.9, 115.8],
      [23.4, 117.1],
      [24.05, 118.5],
      [24.45, 118.08]
    ]
  }
];

const riskTone = {
  Low: 'low',
  Medium: 'medium',
  High: 'high'
};

const vesselIcon = new L.DivIcon({
  className: 'vessel-marker',
  html: '<span></span>',
  iconSize: [22, 22],
  iconAnchor: [11, 11]
});

function App() {
  const selectedVoyage = voyages[0];
  const avgConfidence = Math.round(voyages.reduce((sum, item) => sum + item.confidence, 0) / voyages.length);
  const absoluteError = Math.round(voyages.reduce((sum, item) => sum + Math.abs(item.error), 0) / voyages.length);

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Port ETA Control Tower</p>
          <h1>到港 ETA 可视化 Dashboard</h1>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" aria-label="刷新数据">↻</button>
          <button className="primary-button">导出预测报告</button>
        </div>
      </header>

      <section className="kpi-grid">
        <Metric label="监控航次" value="128" trend="+12 今日新增" />
        <Metric label="平均置信度" value={`${avgConfidence}%`} trend="模型 v8" />
        <Metric label="MAE" value={`${absoluteError} min`} trend="-8.4% vs 昨日" />
        <Metric label="高风险延误" value="9" trend="需人工复核" />
      </section>

      <section className="workspace">
        <aside className="voyage-panel">
          <div className="section-heading">
            <span>航次列表</span>
            <strong>实时</strong>
          </div>
          <div className="voyage-list">
            {voyages.map((voyage) => (
              <article className={`voyage-row ${voyage.id === selectedVoyage.id ? 'active' : ''}`} key={voyage.id}>
                <div className="voyage-main">
                  <h2>{voyage.id}</h2>
                  <p>{voyage.vessel}</p>
                </div>
                <span className={`risk-pill ${riskTone[voyage.delayRisk]}`}>{voyage.delayRisk}</span>
                <div className="voyage-meta">
                  <span>{voyage.origin}</span>
                  <span>{voyage.destination}</span>
                </div>
                <div className="progress-track">
                  <span style={{ width: `${voyage.progress}%` }} />
                </div>
              </article>
            ))}
          </div>
        </aside>

        <section className="map-panel">
          <MapContainer center={[27.5, 121.7]} zoom={5} scrollWheelZoom className="eta-map">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {voyages.map((voyage) => (
              <React.Fragment key={voyage.id}>
                <Polyline positions={voyage.route} pathOptions={{ color: routeColor(voyage.delayRisk), weight: 4, opacity: 0.78 }} />
                <Marker position={voyage.position} icon={vesselIcon}>
                  <Popup>
                    <strong>{voyage.id}</strong>
                    <br />
                    ETA 预测：{voyage.predictedEta}
                  </Popup>
                </Marker>
              </React.Fragment>
            ))}
          </MapContainer>
          <div className="map-overlay">
            <span>East Asia Operations</span>
            <strong>4 active routes</strong>
          </div>
        </section>

        <aside className="prediction-panel">
          <div className="section-heading">
            <span>ETA 预测卡片</span>
            <strong>{selectedVoyage.service}</strong>
          </div>
          <PredictionCard voyage={selectedVoyage} />
          <div className="factor-list">
            <Factor label="潮汐窗口" value="可通行" score={88} />
            <Factor label="港口拥堵" value="低" score={72} />
            <Factor label="天气扰动" value="轻微" score={64} />
            <Factor label="历史相似航段" value="31 条" score={91} />
          </div>
        </aside>
      </section>

      <section className="chart-grid">
        <ChartPanel title="误差分析" option={errorOption} />
        <ChartPanel title="ETA 预测趋势" option={trendOption} />
        <ChartPanel title="风险分布" option={riskOption} />
      </section>
    </main>
  );
}

function Metric({ label, value, trend }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{trend}</p>
    </article>
  );
}

function PredictionCard({ voyage }) {
  return (
    <article className="prediction-card">
      <div className="prediction-header">
        <div>
          <span>当前选中</span>
          <h2>{voyage.id}</h2>
        </div>
        <span className={`risk-pill ${riskTone[voyage.delayRisk]}`}>{voyage.delayRisk}</span>
      </div>
      <dl className="prediction-stats">
        <div>
          <dt>计划 ETA</dt>
          <dd>{voyage.eta}</dd>
        </div>
        <div>
          <dt>预测 ETA</dt>
          <dd>{voyage.predictedEta}</dd>
        </div>
        <div>
          <dt>误差</dt>
          <dd>{voyage.error > 0 ? '+' : ''}{voyage.error} min</dd>
        </div>
        <div>
          <dt>航速</dt>
          <dd>{voyage.speed} kn</dd>
        </div>
      </dl>
      <div className="confidence">
        <div>
          <span>置信度</span>
          <strong>{voyage.confidence}%</strong>
        </div>
        <div className="progress-track">
          <span style={{ width: `${voyage.confidence}%` }} />
        </div>
      </div>
    </article>
  );
}

function Factor({ label, value, score }) {
  return (
    <div className="factor-item">
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div className="mini-bar">
        <span style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

function ChartPanel({ title, option }) {
  return (
    <article className="chart-panel">
      <div className="section-heading">
        <span>{title}</span>
      </div>
      <ReactECharts option={option} className="chart" />
    </article>
  );
}

function routeColor(risk) {
  if (risk === 'High') return '#d95d39';
  if (risk === 'Medium') return '#d6a21f';
  return '#118a7e';
}

const chartText = {
  color: '#243447',
  fontFamily: 'Inter, "Microsoft YaHei", "PingFang SC", sans-serif'
};

const errorOption = {
  color: ['#118a7e', '#d95d39'],
  tooltip: { trigger: 'axis' },
  grid: { left: 34, right: 16, top: 28, bottom: 30 },
  xAxis: {
    type: 'category',
    data: voyages.map((item) => item.id.replace('COSCO-', '')),
    axisLabel: { ...chartText, fontSize: 11 },
    axisLine: { lineStyle: { color: '#c9d5de' } }
  },
  yAxis: {
    type: 'value',
    name: 'min',
    axisLabel: chartText,
    splitLine: { lineStyle: { color: '#e6edf2' } }
  },
  series: [
    {
      name: 'ETA 误差',
      type: 'bar',
      barWidth: 22,
      data: voyages.map((item) => item.error),
      itemStyle: {
        borderRadius: [4, 4, 0, 0],
        color: (params) => (params.value > 60 ? '#d95d39' : '#118a7e')
      }
    }
  ]
};

const trendOption = {
  color: ['#2667ff', '#118a7e'],
  tooltip: { trigger: 'axis' },
  legend: {
    top: 0,
    right: 0,
    textStyle: chartText
  },
  grid: { left: 36, right: 18, top: 38, bottom: 30 },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    axisLabel: chartText,
    axisLine: { lineStyle: { color: '#c9d5de' } }
  },
  yAxis: {
    type: 'value',
    name: 'h',
    axisLabel: chartText,
    splitLine: { lineStyle: { color: '#e6edf2' } }
  },
  series: [
    {
      name: '预测剩余',
      type: 'line',
      smooth: true,
      areaStyle: { opacity: 0.12 },
      data: [42, 37, 30, 24, 18, 12]
    },
    {
      name: '计划剩余',
      type: 'line',
      smooth: true,
      data: [43, 38, 31, 25, 20, 14]
    }
  ]
};

const riskOption = {
  color: ['#118a7e', '#d6a21f', '#d95d39'],
  tooltip: { trigger: 'item' },
  series: [
    {
      type: 'pie',
      radius: ['48%', '72%'],
      center: ['50%', '54%'],
      label: {
        formatter: '{b}\n{d}%',
        color: '#243447',
        fontSize: 12
      },
      data: [
        { name: '低风险', value: 84 },
        { name: '中风险', value: 35 },
        { name: '高风险', value: 9 }
      ]
    }
  ]
};

createRoot(document.getElementById('root')).render(<App />);
