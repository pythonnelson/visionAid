import type { DetectedObject, DetectionEvent } from "./VisionAidObjectDetector/src"

/**
 * Decides WHAT the app says about stable detections. Pure TypeScript (no native calls) so it can be unit tested.
 * The native side only provides the audio (`speak(text)`); all wording lives here so another language
 * (e.g. Krio) can be added by adding one entry to PHRASES / LABELS - nothing else has to change.
 */

export type Direction = 'left' | 'ahead' | 'right'
export type Language = 'en'

/** Words for the three directions. Add 'kri' (Krio) here once reviewed by a speaker. */
const PHRASES: Record<Language, Record<Direction, string>> = {
  en: { left: 'on your left', ahead: 'ahead', right: 'on your right' },
}

/** Optional label translations (identity for English). */
const LABELS: Record<Language, Record<string, string>> = { en: {} }

/** Labels announced first because they matter most when moving around (order = priority). */
export const PRIORITY_LABELS = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train', 'dog', 'stop sign', 'traffic light']

/** Left / ahead / right from the horizontal centre of the box (0 = left edge of the upright image). */
export function directionOf(o: Pick<DetectedObject, 'left' | 'right'>): Direction {
  const cx = (o.left + o.right) / 2
  if (cx < 0.33) return 'left'
  if (cx > 0.67) return 'right'
  return 'ahead'
}

export function describe(o: DetectedObject, lang: Language = 'en'): string {
  const label = LABELS[lang][o.label] ?? o.label
  return `${label} ${PHRASES[lang][directionOf(o)]}`
}

export interface AnnouncerOptions {
  /** The same kind of object is not announced again within this time (stops repeats when detection flickers). */
  cooldownMs?: number
  /** At most this many objects per spoken sentence. */
  maxItems?: number
  /** Injected clock for tests. */
  now?: () => number
  language?: Language
}

export class ObjectAnnouncer {
  private readonly cooldownMs: number
  private readonly maxItems: number
  private readonly now: () => number
  private readonly language: Language
  private lastSpoken = new Map<number, number>()   // classId -> time

  constructor(opts: AnnouncerOptions = {}) {
    this.cooldownMs = opts.cooldownMs ?? 10000
    this.maxItems = opts.maxItems ?? 3
    this.now = opts.now ?? Date.now
    this.language = opts.language ?? 'en'
  }

  /** Forget what was said (use when the mode changes or the camera restarts). */
  reset(): void { this.lastSpoken.clear() }

  /**
   * Turns the newly drained events into ONE sentence (or null when there is nothing new to say).
   * Only 'appeared' events are spoken; objects that disappeared are never announced.
   */
  handle(events: DetectionEvent[]): string | null {
    const t = this.now()
    const fresh = events
      .filter(e => e.type === 'appeared')
      .map(e => e.item)
      .filter(o => {
        const last = this.lastSpoken.get(o.classId)
        return last === undefined || t - last >= this.cooldownMs
      })
    if (fresh.length === 0) return null

    // one entry per class (keep the most confident), most important first, then most confident
    const byClass = new Map<number, DetectedObject>()
    for (const o of fresh) {
      const cur = byClass.get(o.classId)
      if (!cur || o.score > cur.score) byClass.set(o.classId, o)
    }
    const rank = (o: DetectedObject) => {
      const i = PRIORITY_LABELS.indexOf(o.label)
      return i === -1 ? PRIORITY_LABELS.length : i
    }
    const chosen = [...byClass.values()].sort((a, b) => rank(a) - rank(b) || b.score - a.score).slice(0, this.maxItems)
    for (const o of chosen) this.lastSpoken.set(o.classId, t)
    const text = chosen.map(o => describe(o, this.language)).join(', ')
    return text.charAt(0).toUpperCase() + text.slice(1)
  }
}
