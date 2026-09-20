import type { HybridObject } from "react-native-nitro-modules";
import type { Frame } from "react-native-vision-camera";

export interface VisionAidObjectDetector
  extends HybridObject<{
    android: 'kotlin',
    ios: 'swift'
  }> {
    detect(frame: Frame): void
  }
