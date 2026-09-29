import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, ComposedChart
} from 'recharts';
import { 
  TrendingDown, TrendingUp, AlertTriangle, CheckCircle, 
  Map, Calendar, Settings, Activity
} from 'lucide-react';
import './App.css';

// Mock Data for Manganese Reserves and Production
const productionData = [
  { month: 'Jan', target: 45000, actual: 44000, reserveEstimate: 2100000, weatherImpact: 2 },
  { month: 'Feb', target: 45000, actual: 46000, reserveEstimate: 2050000, weatherImpact: 1 },
  { month: 'Mar', target: 48000, actual: 47500, reserveEstimate: 2000000, weatherImpact: 3 },
  { month: 'Apr', target: 48000, actual: 45000, reserveEstimate: 1950000, weatherImpact: 8 }, // Equipment downtime + Rain
  { month: 'May', target: 50000, actual: 41000, reserveEstimate: 1900000, weatherImpact: 15 }, // Blasting delays
  { month: 'Jun', target: 50000, actual: 49000, reserveEstimate: 1850000, weatherImpact: 5 },
];

const riskAlerts = [
  { 
    id: 1, 
    type: 'danger', 
    title: 'Predicted Shortfall (Balaghat Mine)', 
    desc: 'Expected 12% drop due to upcoming heavy rainfall and soil moisture increase.',
    action: 'Adjust Schedule' 
  },
  { 
    id: 2, 
    type: 'warning', 
    title: 'Equipment Maintenance Due', 
    desc: 'Excavator EX-04 at Dongri Buzurg requires maintenance in 3 days. Potential downtime: 48 hrs.',
    action: 'Deploy Backup' 
  },
  { 
    id: 3, 
    type: 'success', 
    title: 'New Reserve Identified', 
    desc: 'Satellite and sub-surface data confirm 50,000 MT additional reserve at Ukwa.',
    action: 'View Map' 
  },
];

function App() {
  return (
    <div className="app-container">
      <header className="header glass-panel">
        <div className="header-title">
          <Activity size={32} />
          <h1>MOIL Manganese Intelligence Dashboard</h1>
        </div>
        <button className="action-btn">
          <Settings size={20} />
          Dashboard Settings
        </button>
      </header>

      <main className="dashboard-grid">
        {/* KPI Row */}
        <div className="kpi-cards">
          <div className="kpi-card glass-panel animate-fade-in" style={{animationDelay: '0.1s'}}>
            <div className="kpi-icon-wrapper primary">
              <Map size={28} />
            </div>
            <div className="kpi-details">
              <span className="kpi-label">Total Est. Reserves</span>
              <span className="kpi-value">1.85M MT</span>
            </div>
          </div>
          <div className="kpi-card glass-panel animate-fade-in" style={{animationDelay: '0.2s'}}>
            <div className="kpi-icon-wrapper success">
              <TrendingUp size={28} />
            </div>
            <div className="kpi-details">
              <span className="kpi-label">YTD Production</span>
              <span className="kpi-value">272,500 MT</span>
            </div>
          </div>
          <div className="kpi-card glass-panel animate-fade-in" style={{animationDelay: '0.3s'}}>
            <div className="kpi-icon-wrapper warning">
              <AlertTriangle size={28} />
            </div>
            <div className="kpi-details">
              <span className="kpi-label">Predicted Shortfall Risk</span>
              <span className="kpi-value">High (12%)</span>
            </div>
          </div>
          <div className="kpi-card glass-panel animate-fade-in" style={{animationDelay: '0.4s'}}>
            <div className="kpi-icon-wrapper danger">
              <Calendar size={28} />
            </div>
            <div className="kpi-details">
              <span className="kpi-label">Blasting Delays (Monthly)</span>
              <span className="kpi-value">4 Days</span>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="chart-section glass-panel animate-fade-in" style={{animationDelay: '0.5s'}}>
          <h2>Production Target vs Actual & Weather Impact</h2>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={productionData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="month" stroke="#94a3b8" />
              <YAxis yAxisId="left" stroke="#94a3b8" />
              <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="target" fill="#6366f1" name="Target (MT)" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="left" dataKey="actual" fill="#ec4899" name="Actual (MT)" radius={[4, 4, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="weatherImpact" stroke="#f59e0b" strokeWidth={3} name="Weather Impact Factor" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* AI Insights & Actions */}
        <div className="side-panel glass-panel animate-fade-in" style={{animationDelay: '0.6s'}}>
          <div className="panel-section">
            <h2>AI-Driven Insights & Corrective Actions</h2>
            <ul className="alert-list">
              {riskAlerts.map((alert) => (
                <li key={alert.id} className={`alert-item ${alert.type}`}>
                  <div className="alert-icon">
                    {alert.type === 'danger' && <AlertTriangle size={24} color="var(--danger)" />}
                    {alert.type === 'warning' && <AlertTriangle size={24} color="var(--warning)" />}
                    {alert.type === 'success' && <CheckCircle size={24} color="var(--success)" />}
                  </div>
                  <div className="alert-content">
                    <h4>{alert.title}</h4>
                    <p>{alert.desc}</p>
                    <button className="action-btn" style={{marginTop: '0.5rem', padding: '0.5rem 1rem', fontSize: '0.875rem'}}>
                      {alert.action}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
        
        {/* Reserve Mapping Chart */}
        <div className="chart-section glass-panel animate-fade-in" style={{animationDelay: '0.7s', gridColumn: 'span 12'}}>
          <h2>Estimated Manganese Reserve Depletion Trend</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={productionData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <defs>
                <linearGradient id="colorReserve" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="month" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={['dataMin - 100000', 'dataMax + 100000']} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)' }}
              />
              <Area type="monotone" dataKey="reserveEstimate" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorReserve)" name="Est. Reserves (MT)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

      </main>
    </div>
  );
}

export default App;
