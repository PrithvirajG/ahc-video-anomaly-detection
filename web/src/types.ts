// Mirrors the payloads in pipeline/api.py. The twelve class strings are the ones
// the dataset's ground_truth.csv uses - they must match exactly or scoring breaks.

export const CLASSES = [
  'normal',
  'traffic_accident',
  'traffic_congestion',
  'stalled_or_broken_down_vehicle',
  'vehicle_blocking_traffic',
  'wrong_way_driving',
  'road_spill_or_debris',
  'waterlogging_or_flood',
  'fire',
  'smoke',
  'fighting_or_violence',
  'loitering_or_suspicious_presence',
] as const

export type ClassName = (typeof CLASSES)[number]

/** One confirmed detection: what fired, where in the video, and why. */
export interface Alert {
  id: string
  video_id: string
  class_name: ClassName
  confidence: number
  start_time_sec: number
  end_time_sec: number | null
  /** The stage-2 VLM's explanation. Blank while only stage 1 has seen it. */
  description_summary: string
  /** Which cascade stage produced this - useful for tuning the filter's recall. */
  stage: 'filter' | 'vlm'
  wall_clock: string
}

/** Per-frame scores from the always-on stage, for the timeline strip. */
export interface ScorePoint {
  t: number
  /** Cerberus-style health score: positive = matches learned normal, negative = deviates. */
  health: number
  anomaly: number
}

export interface Stats {
  fps: number
  frames_seen: number
  frames_escalated: number
  alerts: number
  /** Fraction of frames the cheap stage passed up. Cerberus keeps this near 0.05. */
  escalation_rate: number
  gpu: string
}
