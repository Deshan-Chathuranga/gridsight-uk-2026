import { useState, useEffect, useRef } from "react";
import { 
  ResponsiveContainer, AreaChart, Area, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ReferenceLine
} from "recharts";
import { 
  Sun, RefreshCw, Layers, ShieldAlert, Cpu, 
  Terminal, Calendar, Play, CheckCircle2, XCircle, Clock,
  Cloud, Thermometer, Wind, Info, Moon, Menu, X
} from "lucide-react";

// Define base API URL (empty string represents relative URL in production)
const API_BASE = import.meta.env.VITE_API_URL || "";

// Types definition
interface ForecastPoint {
  timestamp_utc: string;
  y_true_mw: number;
  q10_mw: number;
  q50_mw: number;
  q90_mw: number;
  neso_mw: number;
  capacity_mwp: number;
  is_daylight: boolean;
  ssrd?: number;
  t2m?: number;
  tcc?: number;
  ws10?: number;
  range_mw?: [number, number];
}

interface MetricStats {
  mean_pinball: number;
  coverage_80: number;
  skill_vs_neso: number;
  crossing_rate: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

interface FeatureContribution {
  feature: string;
  contribution: number;
  value: number | null;
  description: string;
}

interface LocalXaiResult {
  base_value: number;
  prediction: number;
  contributions: FeatureContribution[];
}

// Interactive Metric Info Tooltip Explainer Component
function InfoTooltip({ text }: { text: string }) {
  return (
    <div className="info-tooltip-wrapper">
      <span className="info-tooltip-trigger" aria-label="info-tooltip" style={{ cursor: 'pointer' }}>
        <Info size={14} style={{ display: 'inline', marginLeft: '6px', verticalAlign: 'middle' }} />
      </span>
      <div className="info-tooltip-content">
        {text}
      </div>
    </div>
  );
}

export default function App() {
  // State variables
  const [model, setModel] = useState<string>("model_a");
  const [horizon, setHorizon] = useState<number>(24);
  const [split, setSplit] = useState<string>("test");
  const [activeTab, setActiveTab] = useState<string>("analytics");
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const [isAdmin, setIsAdmin] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window !== "undefined" && window.matchMedia) {
      const media = window.matchMedia("(max-width: 1024px)");
      setIsMobile(media.matches);
      const listener = (e: MediaQueryListEvent) => setIsMobile(e.matches);
      media.addEventListener("change", listener);
      return () => media.removeEventListener("change", listener);
    }
  }, []);

  // Parse URL query parameter for admin mode on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      setIsAdmin(params.get("admin") === "true");
    }
  }, []);

  // Fallback: If active tab is pipeline but user is not admin, fallback to analytics view
  useEffect(() => {
    if (activeTab === "pipeline" && !isAdmin) {
      setActiveTab("analytics");
    }
  }, [activeTab, isAdmin]);

  // Theme state persisted in LocalStorage
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    return (localStorage.getItem("gridsight-theme") as "light" | "dark") || "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("gridsight-theme", theme);
  }, [theme]);
  
  const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
  const [filteredData, setFilteredData] = useState<ForecastPoint[]>([]);
  const [metrics, setMetrics] = useState<MetricStats>({
    mean_pinball: 0,
    coverage_80: 0,
    skill_vs_neso: 0,
    crossing_rate: 0
  });
  const [isMock, setIsMock] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  
  // Date Filtering
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({ start: "", end: "" });
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  
  // Explainable AI (XAI)
  const [selectedPoint, setSelectedPoint] = useState<ForecastPoint | null>(null);
  const [globalXai, setGlobalXai] = useState<FeatureImportance[]>([]);
  const [localXai, setLocalXai] = useState<LocalXaiResult | null>(null);
  const [loadingLocalXai, setLoadingLocalXai] = useState<boolean>(false);
  const [metaWeights, setMetaWeights] = useState<any>(null);
  
  // XAI What-If Simulation State
  const [whatifEnabled, setWhatifEnabled] = useState<boolean>(false);
  const [whatifSsrd, setWhatifSsrd] = useState<number>(400);
  const [whatifTcc, setWhatifTcc] = useState<number>(50);
  const [whatifT2m, setWhatifT2m] = useState<number>(15);
  const [simulatedXai, setSimulatedXai] = useState<LocalXaiResult | null>(null);
  const [loadingSimXai, setLoadingSimXai] = useState<boolean>(false);
  
  // Ingestion Pipeline Status & Logs
  const [pipelineStatus, setPipelineStatus] = useState<any>(null);
  const [logs, setLogs] = useState<string>("");
  const [logOffset, setLogOffset] = useState<number>(0);
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // 1. Fetch Forecast Data
  const fetchForecasts = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/forecasts?model=${model}&horizon=${horizon}&split=${split}`
      );
      const res = await response.json();
      if (res.status === "success") {
        const processedData = res.data.map((p: any) => ({
          ...p,
          range_mw: [p.q10_mw, p.q90_mw]
        }));
        setForecastData(processedData);
        setMetrics(res.metrics);
        setIsMock(res.is_mock);
        
        // Extract unique days for sliders/selectors
        if (res.data.length > 0) {
          const dates = Array.from(
            new Set(res.data.map((p: any) => p.timestamp_utc.split("T")[0]))
          ) as string[];
          setAvailableDates(dates);
          // Set default view to the first 4 days for readability
          setDateRange({
            start: dates[0],
            end: dates[Math.min(dates.length - 1, 3)]
          });
        }
      }
    } catch (e) {
      console.error("Error fetching forecasts:", e);
    } finally {
      setLoading(false);
    }
  };

  // 2. Fetch Global XAI (Feature importances and meta coefficients)
  const fetchGlobalXai = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/xai/global?horizon=${horizon}`);
      const res = await response.json();
      if (res.status === "success") {
        setGlobalXai(res.importances);
      }
    } catch (e) {
      console.error("Error fetching global XAI:", e);
    }
    
    try {
      const response = await fetch(`${API_BASE}/api/xai/meta?horizon=${horizon}`);
      const res = await response.json();
      if (res.status === "success") {
        setMetaWeights(res);
      }
    } catch (e) {
      console.error("Error fetching meta weights:", e);
    }
  };

  // 3. Fetch Local SHAP contributions for a clicked timestamp
  const fetchLocalXai = async (timestamp: string) => {
    setLoadingLocalXai(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/xai/local?timestamp=${encodeURIComponent(timestamp)}&horizon=${horizon}`
      );
      const res = await response.json();
      if (res.status === "success") {
        setLocalXai(res);
      }
    } catch (e) {
      console.error("Error fetching local XAI:", e);
    } finally {
      setLoadingLocalXai(false);
    }
  };

  // 3b. Fetch Simulated Local SHAP contributions (What-If override)
  const fetchSimulatedXai = async (timestamp: string, ssrdVal: number, tccVal: number, t2mVal: number) => {
    setLoadingSimXai(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/xai/simulate?timestamp=${encodeURIComponent(timestamp)}&horizon=${horizon}&ssrd=${ssrdVal}&tcc=${tccVal}&t2m=${t2mVal}`
      );
      const res = await response.json();
      if (res.status === "success") {
        setSimulatedXai(res);
      }
    } catch (e) {
      console.error("Error fetching simulated XAI:", e);
    } finally {
      setLoadingSimXai(false);
    }
  };

  // Keep a ref of isSyncing to avoid stale closures in polling interval
  const isSyncingRef = useRef(isSyncing);
  useEffect(() => {
    isSyncingRef.current = isSyncing;
  }, [isSyncing]);

  // 4. Fetch Pipeline Status
  const fetchPipelineStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/pipeline/status`);
      const res = await response.json();
      if (res.status === "success") {
        setPipelineStatus(res);
        const running = res.pipeline_state.status === "RUNNING";
        
        // Trigger data refetch when pipeline sync completes
        if (isSyncingRef.current && !running) {
          fetchForecasts();
          fetchGlobalXai();
        }
        
        setIsSyncing(running);
        if (running) {
          fetchPipelineLogs();
        }
      }
    } catch (e) {
      console.error("Error fetching pipeline status:", e);
    }
  };

  // 5. Fetch Ingestion Logs (Incremental offset support)
  const fetchPipelineLogs = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/pipeline/logs?offset=${logOffset}`);
      const res = await response.json();
      if (res.status === "success") {
        if (res.logs) {
          setLogs((prev) => prev + res.logs);
        }
        setLogOffset(res.offset);
      }
    } catch (e) {
      console.error("Error fetching pipeline logs:", e);
    }
  };

  // 6. Trigger Pipeline Sync manually
  const triggerPipelineSync = async () => {
    setLogs("");
    setLogOffset(0);
    try {
      const response = await fetch(`${API_BASE}/api/pipeline/sync?horizon_steps=${horizon === 6 ? 12 : horizon === 12 ? 24 : 48}`, {
        method: "POST"
      });
      const res = await response.json();
      if (res.status === "success") {
        setIsSyncing(true);
        fetchPipelineStatus();
      } else {
        alert(res.message);
      }
    } catch (e) {
      console.error("Error triggering sync:", e);
    }
  };

  // Filter forecasts locally based on the selected dates
  useEffect(() => {
    if (forecastData.length > 0 && dateRange.start && dateRange.end) {
      const startLimit = `${dateRange.start}T00:00:00`;
      const endLimit = `${dateRange.end}T23:59:59`;
      const filtered = forecastData.filter(
        (p) => p.timestamp_utc >= startLimit && p.timestamp_utc <= endLimit
      );
      setFilteredData(filtered);
      
      // Auto-select the first daytime peak in the window as default local explanation
      const peaks = filtered.filter(p => p.is_daylight && p.y_true_mw > 0);
      if (peaks.length > 0 && !selectedPoint) {
        const sorted = [...peaks].sort((a, b) => b.y_true_mw - a.y_true_mw);
        setSelectedPoint(sorted[0]);
      }
    }
  }, [forecastData, dateRange]);

  // Trigger XAI fetches when model dependencies change
  useEffect(() => {
    fetchForecasts();
  }, [model, horizon, split]);

  useEffect(() => {
    fetchGlobalXai();
  }, [horizon]);

  useEffect(() => {
    if (selectedPoint) {
      fetchLocalXai(selectedPoint.timestamp_utc);
      setWhatifSsrd(selectedPoint.ssrd ?? 0);
      setWhatifTcc(selectedPoint.tcc ?? 0);
      setWhatifT2m(selectedPoint.t2m ?? 0);
      setSimulatedXai(null);
    }
  }, [selectedPoint, horizon]);

  useEffect(() => {
    if (whatifEnabled && selectedPoint) {
      const delayDebounce = setTimeout(() => {
        fetchSimulatedXai(selectedPoint.timestamp_utc, whatifSsrd, whatifTcc, whatifT2m);
      }, 250);
      return () => clearTimeout(delayDebounce);
    }
  }, [whatifEnabled, whatifSsrd, whatifTcc, whatifT2m, selectedPoint, horizon]);

  // Ingestion logs polling
  useEffect(() => {
    fetchPipelineStatus();
    const statusInterval = setInterval(fetchPipelineStatus, 3000);
    return () => clearInterval(statusInterval);
  }, []);

  useEffect(() => {
    let logsInterval: any;
    if (isSyncing) {
      logsInterval = setInterval(fetchPipelineLogs, 1500);
    }
    return () => clearInterval(logsInterval);
  }, [isSyncing, logOffset]);

  // Auto-scroll terminal to end when logs update
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  // Formatter for date ticks
  const formatXAxis = (tick: any): string => {
    try {
      const d = new Date(tick);
      return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}`;
    } catch {
      return String(tick);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "SUCCESS": return "var(--success-green)";
      case "FAILED": return "var(--danger-red)";
      case "RUNNING": return "var(--solar-gold)";
      default: return "var(--text-muted)";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "SUCCESS": return <CheckCircle2 size={16} className="text-emerald-400" style={{ color: "var(--success-green)" }} />;
      case "FAILED": return <XCircle size={16} className="text-red-400" style={{ color: "var(--danger-red)" }} />;
      case "RUNNING": return <RefreshCw size={16} className="animate-spin text-amber-400" style={{ color: "var(--solar-gold)" }} />;
      default: return <Clock size={16} style={{ color: "var(--text-muted)" }} />;
    }
  };

  return (
    <div className="app-container">
      {/* Mobile Top Header Bar */}
      {isMobile && (
        <header className="mobile-header">
          <div className="logo-group">
            <Sun className="logo-icon" />
            <span className="logo-text">GridSight UK</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button 
              className="theme-toggle-btn"
              onClick={() => setTheme(prev => prev === "dark" ? "light" : "dark")}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button 
              className="theme-toggle-btn"
              onClick={() => setSidebarOpen(prev => !prev)}
              title="Toggle settings panel"
              aria-label="Toggle settings panel"
            >
              {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
            </button>
          </div>
        </header>
      )}

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)}></div>
      )}

      {/* SIDEBAR CONTROL PANEL */}
      <aside className={`sidebar ${sidebarOpen ? "mobile-open" : ""}`}>
        <div className="logo-container">
          <div className="logo-group">
            <Sun className="logo-icon" />
            <span className="logo-text">GridSight UK</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button 
              className="theme-toggle-btn"
              onClick={() => setTheme(prev => prev === "dark" ? "light" : "dark")}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button 
              className="theme-toggle-btn mobile-close-btn"
              onClick={() => setSidebarOpen(false)}
              title="Close settings panel"
              aria-label="Close settings panel"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="nav-tabs" aria-label="Sidebar Navigation">
          <button 
            className={`nav-tab-btn ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => { setActiveTab("analytics"); setSidebarOpen(false); }}
          >
            <Layers size={16} />
            <span>Forecast Analytics</span>
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === "xai" ? "active" : ""}`}
            onClick={() => { setActiveTab("xai"); setSidebarOpen(false); }}
          >
            <Cpu size={16} />
            <span>XAI Diagnostics</span>
          </button>
          {isAdmin && (
            <button 
              className={`nav-tab-btn ${activeTab === "pipeline" ? "active" : ""}`}
              onClick={() => { setActiveTab("pipeline"); setSidebarOpen(false); }}
            >
              <Terminal size={16} />
              <span>Data Pipeline</span>
            </button>
          )}
        </nav>
        
        {/* Model Config Section */}
        <div className="sidebar-section">
          <span className="sidebar-section-title">Model Settings</span>
          <div className="control-group">
            <label className="control-label">Forecasting Model</label>
            <select 
              className="select-control" 
              value={model} 
              onChange={(e) => { setModel(e.target.value); setSelectedPoint(null); }}
            >
              <option value="model_a">Model A (Stacking Regressor)</option>
              <option value="model_b">Model B (Standalone LSTM-Q)</option>
              <option value="model_c">Model C (Pretrained Chronos-Q)</option>
            </select>
          </div>

          <div className="control-group">
            <label className="control-label">Forecast Horizon</label>
            <select 
              className="select-control" 
              value={horizon} 
              onChange={(e) => { setHorizon(Number(e.target.value)); setSelectedPoint(null); }}
            >
              <option value={6}>6 Hours Ahead (12 steps)</option>
              <option value={12}>12 Hours Ahead (24 steps)</option>
              <option value={24}>24 Hours Ahead (48 steps)</option>
            </select>
          </div>

          <div className="control-group">
            <label className="control-label">Data Partition Split</label>
            <select 
              className="select-control" 
              value={split} 
              onChange={(e) => { setSplit(e.target.value); setSelectedPoint(null); }}
            >
              <option value="live">Live Forecast (Today/Tomorrow)</option>
              <option value="test">Test Set (Out of Sample)</option>
              <option value="val">Validation Set</option>
            </select>
          </div>
        </div>

        {/* Date Filters Section */}
        {availableDates.length > 0 && (
          <div className="sidebar-section">
            <span className="sidebar-section-title">Time Filter</span>
            <div className="control-group">
              <label className="control-label"><Calendar size={12} /> Start Date</label>
              <select 
                className="select-control"
                value={dateRange.start}
                onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
              >
                {availableDates.map(d => (
                  <option key={d} value={d} disabled={d > dateRange.end}>{d}</option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label"><Calendar size={12} /> End Date</label>
              <select 
                className="select-control"
                value={dateRange.end}
                onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
              >
                {availableDates.map(d => (
                  <option key={d} value={d} disabled={d < dateRange.start}>{d}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </aside>

      {/* MAIN DASHBOARD PANEL */}
      <main className="main-content">
        {/* Mobile View Selector Tab Bar */}
        <div className="mobile-tabs-bar" aria-label="Mobile Navigation tabs">
          <button 
            className={`mobile-tab-btn ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => setActiveTab("analytics")}
          >
            <Layers size={14} />
            <span>Analytics</span>
          </button>
          <button 
            className={`mobile-tab-btn ${activeTab === "xai" ? "active" : ""}`}
            onClick={() => setActiveTab("xai")}
          >
            <Cpu size={14} />
            <span>XAI</span>
          </button>
          {isAdmin && (
            <button 
              className={`mobile-tab-btn ${activeTab === "pipeline" ? "active" : ""}`}
              onClick={() => setActiveTab("pipeline")}
            >
              <Terminal size={14} />
              <span>Pipeline</span>
            </button>
          )}
        </div>

        {/* API Warning if loading Mock */}
        {isMock && (
          <div className="glass-panel" style={{ borderLeft: '4px solid var(--warning-orange)', display: 'flex', gap: '12px', alignItems: 'center', padding: '12px 20px' }}>
            <ShieldAlert style={{ color: 'var(--warning-orange)' }} />
            <div style={{ fontSize: '13px' }}>
              <strong>Demo Sandbox Mode:</strong> Prediction artifacts for the selected model/horizon are not generated locally yet. Generating mock forecasting diagnostics based on test profiles.
            </div>
          </div>
        )}

        {/* Header Metadata */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700', color: 'var(--text-main)' }}>Probabilistic Solar Forecasting Dashboard</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>
              National Grid PV Generation Calibrated Forecasting 
            </p>
          </div>
          {pipelineStatus && pipelineStatus.pipeline_state.completed_at && (
            <div style={{ textAlign: 'right', fontSize: '12px', color: 'var(--text-muted)' }}>
              <span>Last Data Pipeline Sync:</span>
              <div style={{ color: 'var(--text-main)', fontWeight: 'bold', marginTop: '2px' }}>
                {new Date(pipelineStatus.pipeline_state.completed_at).toLocaleString()}
              </div>
            </div>
          )}
        </header>

        {/* Active Tab View Rendering */}
        {activeTab === "analytics" && (
          <>
            {/* Onboarding Welcome Banner */}
            <div className="welcome-banner">
              <div className="welcome-text">
                <h2>📊 Calibrated Solar Dispatch Analytics</h2>
                <p>Welcome. Adjust your forecasting parameters or select a time partition from the sidebar. Click on any peak or data point in the fan chart below to analyze weather conditions and local feature contributions in the diagnostics views.</p>
              </div>
              <button className="welcome-action-btn" onClick={() => setActiveTab("xai")}>Explore Explainable AI →</button>
            </div>

            {/* KPI Metrics Row */}
            <section className="metrics-grid">
              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <span className="metric-title">Pinball Loss</span>
                  <InfoTooltip text="The loss function used to evaluate probabilistic forecasts. It penalizes under-estimation (q10) and over-estimation (q90) based on target percentiles. Lower values indicate better quantile calibration." />
                </div>
                <span className="metric-value">
                  {metrics.mean_pinball !== null && metrics.mean_pinball !== undefined
                    ? metrics.mean_pinball.toFixed(4)
                    : "N/A"}
                </span>
                <div className="metric-sub">
                  <span>Overall quantiles loss (q10/50/90)</span>
                </div>
              </div>

              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <span className="metric-title">Empirical Coverage</span>
                  <InfoTooltip text="The percentage of actual observations that fell within the predicted 80% confidence interval (q10 to q90). The target is exactly 80.0%." />
                </div>
                <span className="metric-value">
                  {metrics.coverage_80 !== null && metrics.coverage_80 !== undefined
                    ? `${(metrics.coverage_80 * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
                <div className="metric-sub">
                  <span>Target Interval Coverage: </span>
                  {metrics.coverage_80 !== null && metrics.coverage_80 !== undefined ? (
                    <span className={Math.abs(metrics.coverage_80 - 0.8) <= 0.03 ? "metric-delta-positive" : "metric-delta-negative"}>
                      {metrics.coverage_80 >= 0.8 ? "+" : ""}{((metrics.coverage_80 - 0.8) * 100).toFixed(1)}% (Goal: 80%)
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)" }}>N/A</span>
                  )}
                </div>
              </div>

              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <span className="metric-title">Skill vs NESO</span>
                  <InfoTooltip text="The percentage improvement in prediction error compared to the National Grid ESO (operator) baseline forecast. A positive value means our models outperform the operator." />
                </div>
                <span className="metric-value">
                  {metrics.skill_vs_neso !== null && metrics.skill_vs_neso !== undefined
                    ? `${(metrics.skill_vs_neso * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
                <div className="metric-sub">
                  {metrics.skill_vs_neso !== null && metrics.skill_vs_neso !== undefined ? (
                    <span className={metrics.skill_vs_neso > 0 ? "metric-delta-positive" : "metric-delta-negative"}>
                      {metrics.skill_vs_neso > 0 ? "Better than Operator Baseline" : "Underperforming Baseline"}
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)" }}>No baseline available</span>
                  )}
                </div>
              </div>

              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <span className="metric-title">Quantile Crossing</span>
                  <InfoTooltip text="Measures the rate of logical errors where a lower quantile (e.g. q10) exceeds a higher quantile (e.g. q50 or q90). Ideally 0.00%, representing strictly monotonic intervals." />
                </div>
                <span className="metric-value">
                  {metrics.crossing_rate !== null && metrics.crossing_rate !== undefined
                    ? `${(metrics.crossing_rate * 100).toFixed(2)}%`
                    : "N/A"}
                </span>
                <div className="metric-sub">
                  {metrics.crossing_rate !== null && metrics.crossing_rate !== undefined ? (
                    <span className={metrics.crossing_rate === 0 ? "metric-delta-positive" : "metric-delta-negative"}>
                      {metrics.crossing_rate === 0 ? "Strictly Monotonic Intervals" : "Anomalous Quantile Crossing"}
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-muted)" }}>N/A</span>
                  )}
                </div>
              </div>
            </section>

            {/* Main Forecast Fan Chart */}
            <section className="glass-panel" style={{ paddingBottom: '30px' }}>
              <div className="chart-header">
                <div className="chart-title-group">
                  <h2>Probabilistic Solar Generation Forecast (MW)</h2>
                  <span className="chart-subtitle">
                    Shaded band represents the calibrated 80% prediction interval (q10 - q90)
                  </span>
                </div>
                <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                  Tip: Click any peak point on the chart to generate local SHAP explanations below.
                </div>
              </div>

              <div className="chart-container">
                {loading ? (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                    <RefreshCw className="animate-spin" /> Loading forecasts...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart 
                      data={filteredData}
                      onClick={(e: any) => {
                        if (e) {
                          if (e.activePayload && e.activePayload.length > 0) {
                            setSelectedPoint(e.activePayload[0].payload);
                          } else if (e.activeTooltipIndex !== undefined && e.activeTooltipIndex !== null && e.activeTooltipIndex >= 0 && e.activeTooltipIndex < filteredData.length) {
                            setSelectedPoint(filteredData[e.activeTooltipIndex]);
                          } else if (e.activeLabel) {
                            const pt = filteredData.find(d => d.timestamp_utc === e.activeLabel);
                            if (pt) setSelectedPoint(pt);
                          }
                        }
                      }}
                      style={{ cursor: 'pointer' }}
                    >
                      <defs>
                        <linearGradient id="colorInterval" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--solar-gold)" stopOpacity={0.25}/>
                          <stop offset="95%" stopColor="var(--solar-gold)" stopOpacity={0.03}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                      <XAxis 
                        dataKey="timestamp_utc" 
                        tickFormatter={formatXAxis} 
                        stroke="#5a6b86"
                        style={{ fontSize: '11px' }}
                        minTickGap={60}
                      />
                      <YAxis 
                        stroke="#5a6b86"
                        style={{ fontSize: '11px' }}
                        label={{ value: 'Generation (MW)', angle: -90, position: 'insideLeft', fill: '#9fb0c7', style: { fontSize: '12px' } }}
                      />
                      <Tooltip 
                        labelFormatter={formatXAxis}
                        contentStyle={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '8px' }}
                        formatter={(value: any, name: any) => {
                          if (typeof value === 'number') {
                            return [`${value.toFixed(1)} MW`, name];
                          }
                          if (Array.isArray(value)) {
                            return [`${value[0].toFixed(1)} ~ ${value[1].toFixed(1)} MW`, name];
                          }
                          return [value, name];
                        }}
                      />
                      <Legend verticalAlign="top" height={36} />
                      
                      {/* Prediction Interval Shaded Band (shades strictly between q10 and q90) */}
                      <Area 
                        name="80% prediction interval (q10-q90)"
                        type="monotone" 
                        dataKey="range_mw" 
                        stroke="none"
                        fill="url(#colorInterval)"
                      />
                      
                      {/* Bounding curves for the prediction interval */}
                      <Line 
                        name="q90 Upper Bound"
                        type="monotone" 
                        dataKey="q90_mw" 
                        stroke="var(--danger-red)" 
                        strokeWidth={1.2}
                        strokeDasharray="3 3"
                        dot={false}
                        legendType="none"
                      />
                      <Line 
                        name="q10 Lower Bound"
                        type="monotone" 
                        dataKey="q10_mw" 
                        stroke="var(--q10-cyan)" 
                        strokeWidth={1.2}
                        strokeDasharray="3 3"
                        dot={false}
                        legendType="none"
                      />

                      {/* Vertical line indicator at selected timestamp */}
                      {selectedPoint && (
                        <ReferenceLine 
                          x={selectedPoint.timestamp_utc} 
                          stroke="var(--solar-gold)" 
                          strokeWidth={1.5}
                          strokeDasharray="4 4" 
                          label={{ value: "Selected", position: "top", fill: "var(--solar-gold)", fontSize: 10, fontWeight: "bold" }}
                        />
                      )}
                      
                      {/* Median Prediction */}
                      <Line 
                        name="q50 Median Forecast"
                        type="monotone" 
                        dataKey="q50_mw" 
                        stroke="var(--solar-gold)" 
                        strokeWidth={2.5}
                        dot={false}
                      />
                      
                      {/* Actual Generation */}
                      <Line 
                        name="Actual Generation"
                        type="monotone" 
                        dataKey="y_true_mw" 
                        stroke="var(--actual-green)" 
                        strokeWidth={2}
                        dot={false}
                      />
                      
                      {/* NESO Baseline */}
                      <Line 
                        name="NESO operator baseline"
                        type="monotone" 
                        dataKey="neso_mw" 
                        stroke="var(--baseline-magenta)" 
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </section>

            {/* SELECTED POINT DIAGNOSTICS CARD */}
            {selectedPoint ? (
              <section className="glass-panel selection-diagnostics-panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', padding: '20px' }}>
                
                {/* COLUMN 1: QUANTILE & CONFIDENCE INTERVALS DETAILS */}
                <div className="diagnostics-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <h3 style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--text-main)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Layers size={16} style={{ color: 'var(--solar-gold)' }} />
                      Interval & Target Values
                    </h3>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(255,255,255,0.015)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>q90 (Upper Bound):</span>
                        <span style={{ fontWeight: 'bold', color: 'var(--text-main)' }}>{selectedPoint.q90_mw !== null ? `${selectedPoint.q90_mw.toFixed(1)} MW` : "N/A"}</span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(245, 158, 11, 0.04)', border: '1px solid rgba(245, 158, 11, 0.12)', borderRadius: '6px' }}>
                        <span style={{ color: 'var(--solar-gold)', fontWeight: '500' }}>q50 (Expected Median):</span>
                        <span style={{ fontWeight: 'bold', color: 'var(--solar-gold)' }}>{selectedPoint.q50_mw !== null ? `${selectedPoint.q50_mw.toFixed(1)} MW` : "N/A"}</span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(255,255,255,0.015)', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>q10 (Lower Bound):</span>
                        <span style={{ fontWeight: 'bold', color: 'var(--text-main)' }}>{selectedPoint.q10_mw !== null ? `${selectedPoint.q10_mw.toFixed(1)} MW` : "N/A"}</span>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.12)', borderRadius: '6px' }}>
                        <span style={{ color: 'var(--success-green)', fontWeight: '500' }}>Actual Generation:</span>
                        <span style={{ fontWeight: 'bold', color: 'var(--success-green)' }}>
                          {selectedPoint.y_true_mw !== null && selectedPoint.y_true_mw !== undefined ? `${selectedPoint.y_true_mw.toFixed(1)} MW` : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Interval Bounding Assessment */}
                  {selectedPoint.y_true_mw !== null && selectedPoint.y_true_mw !== undefined && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', fontSize: '12px', marginTop: '12px', border: '1px solid var(--border-color)' }}>
                      <Info size={14} style={{ color: 'var(--solar-gold)', flexShrink: 0 }} />
                      <span>
                        {selectedPoint.y_true_mw >= selectedPoint.q10_mw && selectedPoint.y_true_mw <= selectedPoint.q90_mw ? (
                          <span style={{ color: 'var(--success-green)' }}>✓ Actual is within the 80% confidence interval.</span>
                        ) : selectedPoint.y_true_mw > selectedPoint.q90_mw ? (
                          <span style={{ color: 'var(--danger-red)' }}>⚠️ Under-predicted: Actual exceeded the 90th percentile.</span>
                        ) : (
                          <span style={{ color: 'var(--danger-red)' }}>⚠️ Over-predicted: Actual fell below the 10th percentile.</span>
                        )}
                      </span>
                    </div>
                  )}
                </div>

                {/* COLUMN 2: WEATHER CONDITIONS AT CLICKED TIMESTAMP */}
                <div className="diagnostics-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Sun size={16} style={{ color: 'var(--solar-gold)' }} />
                        NWP Weather Diagnostics
                      </h3>
                      
                      <button 
                        onClick={() => { setActiveTab("xai"); setWhatifEnabled(true); }}
                        style={{ background: 'rgba(249, 115, 22, 0.08)', border: '1px solid rgba(249, 115, 22, 0.2)', color: 'var(--warning-orange)', borderRadius: '20px', padding: '4px 10px', fontSize: '11px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s' }}
                      >
                        Try Simulator
                      </button>
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                      <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Sun size={12} style={{ color: 'var(--solar-gold)' }} />
                          Radiation (SSRD)
                        </span>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-main)' }}>
                          {selectedPoint.ssrd !== undefined && selectedPoint.ssrd !== null ? `${selectedPoint.ssrd.toFixed(1)} W/m²` : "N/A"}
                        </span>
                      </div>

                      <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Cloud size={12} style={{ color: 'var(--q10-cyan)' }} />
                          Cloud Cover (TCC)
                        </span>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-main)' }}>
                          {selectedPoint.tcc !== undefined && selectedPoint.tcc !== null ? `${selectedPoint.tcc.toFixed(1)}%` : "N/A"}
                        </span>
                      </div>

                      <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Thermometer size={12} style={{ color: 'var(--warning-orange)' }} />
                          Temperature
                        </span>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-main)' }}>
                          {selectedPoint.t2m !== undefined && selectedPoint.t2m !== null ? `${selectedPoint.t2m.toFixed(1)} °C` : "N/A"}
                        </span>
                      </div>

                      <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Wind size={12} style={{ color: 'var(--text-muted)' }} />
                          Wind Speed
                        </span>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-main)' }}>
                          {selectedPoint.ws10 !== undefined && selectedPoint.ws10 !== null ? `${selectedPoint.ws10.toFixed(1)} m/s` : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Extra Weather Info summary text */}
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '12px', padding: '10px', background: 'rgba(255,255,255,0.01)', borderRadius: '6px', lineHeight: '1.45', border: '1px solid var(--border-color)' }}>
                    {(selectedPoint.ssrd ?? 0) > 100 ? (
                      <span>Optimal solar drivers: high SSRD of {(selectedPoint.ssrd ?? 0).toFixed(0)} W/m² generates solar dispatch capacity.</span>
                    ) : (
                      <span>Weak solar driver: radiation is near zero due to cloud cover ({(selectedPoint.tcc ?? 0).toFixed(0)}%) or night gating.</span>
                    )}
                  </div>
                </div>
                
              </section>
            ) : (
              <div className="glass-panel" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)', borderStyle: 'dashed' }}>
                Click a point on the forecasting chart above to load specific timestamp details.
              </div>
            )}
          </>
        )}

        {/* TAB 2: EXPLAINABLE AI (XAI) DIAGNOSTICS */}
        {activeTab === "xai" && (
          <>
            <div className="welcome-banner">
              <div className="welcome-text">
                <h2>🧠 Explainable AI (XAI) & What-If Simulations</h2>
                <p>Understand the physics-based gating constraints and machine learning feature importances driving predictions. Toggle the What-If simulation below to mock live NWP weather adjustments and immediately calculate LightGBM SHAP outputs.</p>
              </div>
              <button className="welcome-action-btn" onClick={() => setActiveTab("analytics")}>← View Forecast Fan Chart</button>
            </div>

            <div className="diagnostics-grid">
              {/* Local SHAP contributions */}
              <div className="glass-panel">
                <div className="xai-header">
                  <span className="xai-title">Local Explainability (TreeSHAP Contributions)</span>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '12px', color: '#9fb0c7' }}>Timestamp:</span>
                    <select
                      className="select-control"
                      style={{ padding: '6px 12px', fontSize: '12px', height: 'auto', background: 'var(--input-bg)', color: 'var(--text-main)', width: 'auto' }}
                      value={selectedPoint ? selectedPoint.timestamp_utc : ""}
                      onChange={(e) => {
                        const pt = filteredData.find(d => d.timestamp_utc === e.target.value);
                        if (pt) setSelectedPoint(pt);
                      }}
                    >
                      <option value="" disabled>-- Select Timestamp --</option>
                      {filteredData.map(pt => (
                        <option key={pt.timestamp_utc} value={pt.timestamp_utc}>
                          {new Date(pt.timestamp_utc).toLocaleString()}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {selectedPoint ? (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', background: 'rgba(255,255,255,0.015)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <span style={{ fontSize: '13px', color: 'var(--text-highlight)' }}>
                        Selected: <strong>{new Date(selectedPoint.timestamp_utc).toLocaleString()}</strong>
                      </span>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12px', background: whatifEnabled ? 'rgba(249, 115, 22, 0.15)' : 'rgba(255,255,255,0.03)', padding: '6px 12px', borderRadius: '20px', border: whatifEnabled ? '1px solid rgba(249, 115, 22, 0.3)' : '1px solid rgba(255,255,255,0.08)', transition: 'all 0.2s', userSelect: 'none' }}>
                        <input 
                          type="checkbox" 
                          checked={whatifEnabled} 
                          onChange={(e) => setWhatifEnabled(e.target.checked)} 
                          style={{ cursor: 'pointer', accentColor: 'var(--warning-orange)' }} 
                        />
                        <span style={{ color: whatifEnabled ? 'var(--warning-orange)' : 'var(--text-muted)', fontWeight: '600' }}>
                          What-If Simulator Mode
                        </span>
                      </label>
                    </div>

                    {whatifEnabled && (
                      <div className="diagnostics-card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px', marginBottom: '18px', border: '1px solid rgba(249, 115, 22, 0.25)', background: 'rgba(249, 115, 22, 0.02)' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Sun size={12} style={{ color: 'var(--solar-gold)' }} /> SSRD: {whatifSsrd.toFixed(0)} W/m²
                          </span>
                          <input 
                            type="range" min="0" max="1000" step="10"
                            value={whatifSsrd} onChange={(e) => setWhatifSsrd(Number(e.target.value))} 
                            style={{ width: '100%', accentColor: 'var(--warning-orange)', cursor: 'ew-resize' }}
                          />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Cloud size={12} style={{ color: 'var(--q10-cyan)' }} /> Cloud Cover: {whatifTcc.toFixed(0)}%
                          </span>
                          <input 
                            type="range" min="0" max="100" step="1"
                            value={whatifTcc} onChange={(e) => setWhatifTcc(Number(e.target.value))} 
                            style={{ width: '100%', accentColor: 'var(--warning-orange)', cursor: 'ew-resize' }}
                          />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Thermometer size={12} style={{ color: 'var(--warning-orange)' }} /> Temp: {whatifT2m.toFixed(1)} °C
                          </span>
                          <input 
                            type="range" min="-10" max="40" step="0.5"
                            value={whatifT2m} onChange={(e) => setWhatifT2m(Number(e.target.value))} 
                            style={{ width: '100%', accentColor: 'var(--warning-orange)', cursor: 'ew-resize' }}
                          />
                        </div>
                      </div>
                    )}

                    {loadingLocalXai || (whatifEnabled && loadingSimXai) ? (
                      <div style={{ display: 'flex', padding: '60px', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <RefreshCw className="animate-spin" /> Querying SHAP contributors...
                      </div>
                    ) : (whatifEnabled ? simulatedXai : localXai) ? (
                      <div className="waterfall-container">
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginBottom: '8px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Feature Name (Actual NWP Value)</span>
                          <span style={{ color: 'var(--text-muted)' }}>SHAP Impact (Capacity Change)</span>
                        </div>
                        {(whatifEnabled ? simulatedXai! : localXai!).contributions.map((c) => {
                          const absVal = Math.min(100, Math.max(1, Math.abs(c.contribution) * 200)); 
                          return (
                            <div className="waterfall-row" key={c.feature}>
                              <span className="waterfall-feat-name" title={c.description}>
                                {c.feature} <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>({c.value !== null ? c.value.toFixed(2) : "N/A"})</span>
                              </span>
                              
                              <div className="waterfall-bar-wrapper">
                                <div 
                                  className={`waterfall-bar ${c.contribution >= 0 ? "positive" : "negative"}`}
                                  style={{ 
                                    width: `${absVal}%`, 
                                    left: c.contribution >= 0 ? '50%' : 'auto', 
                                    right: c.contribution >= 0 ? 'auto' : '50%' 
                                  }}
                                ></div>
                                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', background: 'rgba(255,255,255,0.15)' }}></div>
                              </div>
                              
                              <span className={`waterfall-val ${c.contribution >= 0 ? "positive" : "negative"}`}>
                                {c.contribution >= 0 ? "+" : ""}{c.contribution.toFixed(4)}
                              </span>
                            </div>
                          );
                        })}
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', background: 'rgba(255,255,255,0.015)', padding: '12px', borderRadius: '8px', fontSize: '13px', border: '1px solid var(--border-color)' }}>
                          <span>Model Expected Base: <strong>{(whatifEnabled ? simulatedXai! : localXai!).base_value.toFixed(4)}</strong></span>
                          <span>LGBM Prediction: <strong style={{ color: whatifEnabled ? 'var(--warning-orange)' : 'var(--solar-gold)' }}>{(whatifEnabled ? simulatedXai! : localXai!).prediction.toFixed(4)}</strong></span>
                        </div>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                        Select a timestamp to inspect local TreeSHAP diagnostics.
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)', border: '1px dashed var(--border-color)', borderRadius: '8px' }}>
                    <Cpu size={24} style={{ marginBottom: '8px', color: 'var(--solar-gold)' }} />
                    <span>No forecast timestamp selected. Select one from the header dropdown above.</span>
                  </div>
                )}
              </div>

              {/* Global Feature Importances, Meta weights, Physics gating */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="glass-panel">
                  <div className="xai-header">
                    <span className="xai-title">Global Feature Importance (LGBM)</span>
                  </div>
                  <p className="xai-desc" style={{ marginBottom: '14px' }}>
                    Parameters driving tabular models across all prediction horizons.
                  </p>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {globalXai.slice(0, 5).map((f) => (
                      <div key={f.feature} style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '13px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span><strong>{f.feature}</strong> <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>— {f.description}</span></span>
                          <span style={{ fontFamily: 'monospace', fontWeight: 'bold', color: 'var(--solar-gold)' }}>{f.importance}</span>
                        </div>
                        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px' }}>
                          <div 
                            style={{ 
                              width: `${Math.min(100, (f.importance / Math.max(1, globalXai[0]?.importance || 1)) * 100)}%`, 
                              height: '100%', 
                              background: 'var(--solar-gold)', 
                              borderRadius: '2px' 
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {metaWeights && (
                  <div className="glass-panel">
                    <span className="xai-title" style={{ fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                      <Layers size={14} style={{ color: 'var(--solar-gold)' }} /> Meta-Learner Stacked Quantile Weights
                    </span>
                    <p className="xai-desc" style={{ fontSize: '12px', marginBottom: '10px' }}>
                      Stacking models combine Base Predictions with Clear-Sky GHI.
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '12px', fontSize: '12px', background: 'rgba(0,0,0,0.15)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <div>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Base Quantile Predictors:</div>
                        <div style={{ fontWeight: '500' }}>TCN (q10/50/90)</div>
                        <div style={{ fontWeight: '500' }}>LGBM (q10/50/90)</div>
                        <div style={{ fontWeight: '500' }}>Clear-Sky GHI Gate</div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Primary weights (median q50):</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>LGBM q50:</span>
                            <strong style={{ color: 'var(--solar-gold)' }}>{(metaWeights.weights && metaWeights.weights[0.5] ? metaWeights.weights[0.5][4].toFixed(2) : "0.40")}</strong>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>TCN q50:</span>
                            <strong style={{ color: 'var(--solar-gold)' }}>{(metaWeights.weights && metaWeights.weights[0.5] ? metaWeights.weights[0.5][1].toFixed(2) : "0.35")}</strong>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Clear-Sky GHI:</span>
                            <strong style={{ color: 'var(--solar-gold)' }}>{(metaWeights.weights && metaWeights.weights[0.5] ? metaWeights.weights[0.5][6].toFixed(2) : "0.05")}</strong>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="glass-panel">
                  <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-main)', display: 'block', marginBottom: '8px' }}>
                    ☀️ Physical Night Gating Constraints
                  </span>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    To ensure solar generation complies with laws of astronomy, a physical check is applied: if the solar elevation angle <code style={{ color: 'var(--solar-gold)', background: 'rgba(0,0,0,0.2)', padding: '2px 4px', borderRadius: '4px' }}>&lt; -5.0°</code>, the gating constraints override predictions to exactly <strong>0.00 MW</strong>.
                  </p>
                </div>
              </div>
            </div>
          </>
        )}

        {/* TAB 3: DATA PIPELINE & STORAGE LOGS */}
        {activeTab === "pipeline" && (
          <>
            <div className="welcome-banner">
              <div className="welcome-text">
                <h2>⚙️ National Grid Ingestion Data Pipeline</h2>
                <p>Monitor raw and transformed file storage directories, track database partitions, and execute pipeline sync runs. Check execution logs in real-time below.</p>
              </div>
            </div>

            {/* Storage Health cards */}
            {pipelineStatus && (
              <section className="metrics-grid">
                <div className="glass-panel metric-card">
                  <span className="metric-title">Bronze Raw Storage</span>
                  <span className="metric-value" style={{ color: 'var(--text-main)' }}>
                    {pipelineStatus.storage_stats.bronze.size_mb} MB
                  </span>
                  <div className="metric-sub">
                    <span>{pipelineStatus.storage_stats.bronze.file_count} raw XML GHI feeds loaded</span>
                  </div>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-title">Silver Cleaned Data</span>
                  <span className="metric-value" style={{ color: 'var(--text-main)' }}>
                    {pipelineStatus.storage_stats.silver.size_mb} MB
                  </span>
                  <div className="metric-sub">
                    <span>{pipelineStatus.storage_stats.silver.file_count} hourly structured datasets</span>
                  </div>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-title">Gold Feature Matrix</span>
                  <span className="metric-value" style={{ color: 'var(--text-main)' }}>
                    {pipelineStatus.storage_stats.gold.size_mb} MB
                  </span>
                  <div className="metric-sub">
                    <span>{pipelineStatus.storage_stats.gold.file_count} tabular feature stores</span>
                  </div>
                </div>
              </section>
            )}

            {/* Sync control card and logs */}
            <div className="glass-panel" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <span className="xai-title">Pipeline Control Actions</span>
                <p className="xai-desc" style={{ marginBottom: '10px' }}>
                  Trigger an on-demand sync of weather forecasts and solar capacity factors.
                </p>

                <div className="diagnostics-card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '13px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Daily Sync Status:</span>
                    <span style={{ color: 'var(--success-green)', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--success-green)' }}></span> Active
                    </span>
                  </div>

                  {pipelineStatus && (
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px', borderTop: '1px solid var(--border-color)', paddingTop: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {getStatusIcon(pipelineStatus.pipeline_state.status)}
                        <span>Status: <strong style={{ color: getStatusColor(pipelineStatus.pipeline_state.status) }}>{pipelineStatus.pipeline_state.status}</strong></span>
                      </div>
                      <span>Stage: <strong style={{ color: 'var(--text-main)' }}>{pipelineStatus.pipeline_state.current_stage || "IDLE"}</strong></span>
                      {pipelineStatus.pipeline_state.last_run_timestamp && (
                        <span>Last run: <strong style={{ color: 'var(--text-main)' }}>{pipelineStatus.pipeline_state.last_run_timestamp}</strong></span>
                      )}
                    </div>
                  )}

                  <button 
                    className="btn-primary" 
                    onClick={triggerPipelineSync} 
                    disabled={isSyncing}
                    style={{ marginTop: '10px' }}
                  >
                    <Play size={14} /> {isSyncing ? "Syncing..." : "Sync Ingest Data"}
                  </button>
                </div>
              </div>

              {/* Logs output console */}
              <div>
                <div className="terminal-title-bar">
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Terminal size={14} /> Ingestion Sync Console Output
                  </span>
                  <span>offset: {logOffset} bytes</span>
                </div>
                
                <div className="terminal-panel">
                  {logs ? logs : "Pipeline logs are idle. Trigger a sync request to start logging raw data pipeline transactions..."}
                  <div ref={terminalEndRef} />
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
