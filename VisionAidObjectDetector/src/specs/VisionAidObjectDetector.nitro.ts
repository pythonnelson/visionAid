import type { HybridObject } from "react-native-nitro-modules";
import type { Frame } from "react-native-vision-camera";

/**
 * One detected object. Box coordinates are NORMALISED (0..1) and refer to the UPRIGHT camera image
 * (the frame as seen holding the phone in its natural orientation), origin top-left.
 */
export interface DetectedObject {
  /** Human readable COCO label, e.g. "chair". */
  label: string
  /** Zero-based class index in coco_labels.txt. */
  classId: number
  /** Model score 0..1 (a score, NOT a proven probability of being right). */
  score: number
  left: number
  top: number
  right: number
  bottom: number
}

/** The objects that have been seen consistently over several frames ("stable"), plus how old that information is. */
export interface StableDetections {
  /** Size in pixels of the upright analysed frame the boxes refer to (0 before the first analysed frame). */
  frameWidth: number
  frameHeight: number
  /** Milliseconds since the detector last produced this list; -1 if no frame was analysed yet.
   *  Large values mean the camera is no longer feeding the detector (e.g. currency mode is on). */
  ageMs: number
  detections: DetectedObject[]
}

/** What happened to an object in the stable set. */
export type DetectionEventType = 'appeared' | 'disappeared'

/** A change in the stable set. For 'disappeared' the box is the last known position. */
export interface DetectionEvent {
  type: DetectionEventType
  item: DetectedObject
}

export interface VisionAidObjectDetector
  extends HybridObject<{
    android: 'kotlin',
    ios: 'swift'
  }> {
    /** Analyse one camera frame (unchanged since the first version). */
    detect(frame: Frame): void
    /** Latest stable detections (cheap; safe to poll a few times per second). */
    getStableDetections(): StableDetections
    /** Everything that appeared / disappeared since the last call. The queue is emptied; at most the newest 32 events are kept. */
    drainEvents(): DetectionEvent[]
    /** Forget queued events and the current stable list (call when another mode takes over the camera). */
    reset(): void
    /** Speak `text` with the phone's text-to-speech (English voice). JavaScript decides what to say. */
    speak(text: string, interrupt: boolean): void
    stopSpeaking(): void
  }
