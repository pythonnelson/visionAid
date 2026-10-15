import type { HybridObject } from "react-native-nitro-modules";
import type { Frame } from "react-native-vision-camera";

/**
 * Sierra Leone banknote recogniser (separate model from the object detector).
 * All decisions (confidence, consistency, "I'm not sure", speaking) happen natively;
 * JavaScript only switches the mode on/off and reads the text to display.
 */
export interface VisionAidCurrencyRecognizer
  extends HybridObject<{
    android: 'kotlin',
    ios: 'swift'
  }> {
    /** Turns currency mode on/off. Turning it off stops speech and forgets the last result. */
    setActive(active: boolean): void
    isActive(): boolean
    /** Analyse one camera frame (no-op while inactive; rate limited natively). */
    analyze(frame: Frame): void
    /** Text to show on screen (what the app last said, or a hint). */
    getStatus(): string
    /** Empty string when everything is fine, otherwise why currency recognition is unavailable. */
    getError(): string
  }
