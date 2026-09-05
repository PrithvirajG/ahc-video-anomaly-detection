import { useEffect, useMemo, useRef, useState } from 'react'
import { Area, AreaChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { Alert, ScorePoint, Stats } from './types'
import { CLASSES } from './types'
import './App.css'

const MAX_POINTS = 600 // ~1 minute of timeline at 10 Hz

export default function App() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [scores, setScores] = useState<ScorePoint[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [connected, setConnected] = useState(false)
  const [selected, setSelected] = useState<Alert | null>(null)
  const [muted, setMuted] = useState<Set<string>>(new Set())
  const videoRef = useRef<HTMLVideoElement>(null)

  // Single websocket carries all three message kinds so the UI never polls.
  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/alerts`)

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'alert') setAlerts((a) => [msg.data as Alert, ...a].slice(0, 200))
      else if (msg.type === 'score') setScores((s) => [...s, msg.data as ScorePoint].slice(-MAX_POINTS))
      else if (msg.type === 'stats') setStats(msg.data as Stats)
    }
    return () => ws.close()
  }, [])

  // Jumping to an alert seeks a little before the event so the operator sees the lead-in.
  useEffect(() => {
    if (selected && videoRef.current) {
      videoRef.current.currentTime = Math.max(0, selected.start_time_sec - 2)
      void videoRef.current.play().catch(() => {})
    }
  }, [selected])

  const visible = useMemo(
    () => alerts.filter((a) => !muted.has(a.class_name)),
    [alerts, muted],
  )

  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const a of alerts) c.set(a.class_name, (c.get(a.class_name) ?? 0) + 1)
    return c
  }, [alerts])

  function toggleMute(cls: string) {
    setMuted((m) => {
      const next = new Set(m)
      next.has(cls) ? next.delete(cls) : next.add(cls)
      return next
    })
  }

  return (
    <div className="app">
      <header>
        <h1>AHC · Video Anomaly Detection</h1>
        <div className="stats">
          <Stat label="fps" value={stats ? stats.fps.toFixed(1) : '—'} />
          <Stat
            label="escalated"
            value={stats ? `${(stats.escalation_rate * 100).toFixed(1)}%` : '—'}
            hint="fraction of frames the cheap stage passed to the VLM"
          />
          <Stat label="alerts" value={String(alerts.length)} />
          <Stat label="gpu" value={stats?.gpu ?? '—'} />
          <span className={connected ? 'dot live' : 'dot dead'} title={connected ? 'live' : 'disconnected'} />
        </div>
      </header>

      <main>
        <section className="left">
          <video ref={videoRef} controls src={selected ? `/clips/${selected.video_id}.mp4` : undefined} />

          <div className="timeline">
            <ResponsiveContainer width="100%" height={110}>
              <AreaChart data={scores} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.55} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" tick={{ fontSize: 10 }} stroke="var(--dim)" />
                <YAxis domain={[0, 1]} width={28} tick={{ fontSize: 10 }} stroke="var(--dim)" />
                {/* Where the always-on stage decides to wake the VLM. */}
                <ReferenceLine y={0.5} stroke="var(--warn)" strokeDasharray="3 3" />
                <Area type="monotone" dataKey="anomaly" stroke="var(--accent)" fill="url(#g)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chips">
            {CLASSES.filter((c) => c !== 'normal').map((c) => (
              <button
                key={c}
                className={`chip ${muted.has(c) ? 'off' : ''}`}
                onClick={() => toggleMute(c)}
                title="click to mute this class"
              >
                {c.replace(/_/g, ' ')}
                {counts.get(c) ? <b>{counts.get(c)}</b> : null}
              </button>
            ))}
          </div>
        </section>

        <section className="right">
          <h2>Alerts</h2>
          {visible.length === 0 && (
            <p className="empty">
              Nothing yet. Start the detector:
              <code>uv run uvicorn pipeline.api:app --reload --port 8010</code>
            </p>
          )}
          <ul className="feed">
            {visible.map((a) => (
              <li
                key={a.id}
                className={`alert ${selected?.id === a.id ? 'sel' : ''} stage-${a.stage}`}
                onClick={() => setSelected(a)}
              >
                <div className="row">
                  <span className="cls">{a.class_name.replace(/_/g, ' ')}</span>
                  <span className="conf">{(a.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="row sub">
                  <span>{a.video_id}</span>
                  <span>
                    {a.start_time_sec.toFixed(1)}s
                    {a.end_time_sec !== null && `–${a.end_time_sec.toFixed(1)}s`}
                  </span>
                </div>
                {a.description_summary && <p className="desc">{a.description_summary}</p>}
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat" title={hint}>
      <span className="k">{label}</span>
      <span className="v">{value}</span>
    </div>
  )
}
