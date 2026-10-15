import type { DetectedObject } from "./VisionAidObjectDetector/src"

export interface ViewBox { x: number, y: number, width: number, height: number }

/**
 * Maps a normalised box (0..1, upright camera frame) to pixels inside the preview view, assuming the preview
 * uses VisionCamera's default resizeMode 'cover' (frame scaled until it fills the view; the overflow is cropped
 * equally on both sides). Returns null when the box is entirely outside the visible part or sizes are unknown.
 * Assumes the analysed frame shows the same field of view as the preview (not yet verified on the device).
 */
export function boxToView(
  o: Pick<DetectedObject, 'left' | 'top' | 'right' | 'bottom'>,
  frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number
): ViewBox | null {
  if (!(frameWidth > 0 && frameHeight > 0 && viewWidth > 0 && viewHeight > 0)) return null
  const scale = Math.max(viewWidth / frameWidth, viewHeight / frameHeight)
  const offsetX = (viewWidth - frameWidth * scale) / 2
  const offsetY = (viewHeight - frameHeight * scale) / 2
  const x0 = Math.max(0, o.left * frameWidth * scale + offsetX)
  const y0 = Math.max(0, o.top * frameHeight * scale + offsetY)
  const x1 = Math.min(viewWidth, o.right * frameWidth * scale + offsetX)
  const y1 = Math.min(viewHeight, o.bottom * frameHeight * scale + offsetY)
  if (x1 <= x0 || y1 <= y0) return null
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}
