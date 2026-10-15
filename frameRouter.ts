import type { Frame } from "react-native-vision-camera"
import type { VisionAidObjectDetector, VisionAidCurrencyRecognizer } from "./VisionAidObjectDetector/src"

/**
 * Sends each camera frame to exactly ONE analyser:
 * the currency recogniser while currency mode is on, otherwise the object detector.
 * Called from the VisionCamera frame worklet, so it must stay tiny and synchronous.
 */
export function routeFrame(
  frame: Frame,
  objectDetector: Pick<VisionAidObjectDetector, 'detect'>,
  currency: Pick<VisionAidCurrencyRecognizer, 'isActive' | 'analyze'>
): 'currency' | 'objects' {
  'worklet';
  if (currency.isActive()) {
    currency.analyze(frame)
    return 'currency'
  }
  objectDetector.detect(frame)
  return 'objects'
}
