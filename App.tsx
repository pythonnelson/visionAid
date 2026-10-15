import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { LayoutChangeEvent, StyleSheet, Text, TouchableOpacity, View } from "react-native"
import { Camera, useCameraDevice, useCameraPermission, useFrameOutput } from "react-native-vision-camera"
import { NitroModules } from "react-native-nitro-modules"
import type { VisionAidObjectDetector, VisionAidCurrencyRecognizer, StableDetections } from "./VisionAidObjectDetector/src"
import { routeFrame } from "./frameRouter"
import { ObjectAnnouncer, describe } from "./objectAnnouncer"
import { boxToView } from "./overlayLayout"

/** Stable detections older than this are ignored (the camera is no longer feeding the detector). */
const MAX_DETECTION_AGE_MS = 1500

function App(): React.JSX.Element {
  const device = useCameraDevice("back")
  const { hasPermission, requestPermission } = useCameraPermission()

  useEffect(() => {
    if (!hasPermission) {
      requestPermission()
    }
  }, [hasPermission, requestPermission])

  const objectDetector = useMemo(
      () => NitroModules.createHybridObject<VisionAidObjectDetector>('VisionAidObjectDetector'),
      []
    )
  const currencyRecognizer = useMemo(
      () => NitroModules.createHybridObject<VisionAidCurrencyRecognizer>('VisionAidCurrencyRecognizer'),
      []
    )
  const [currencyOn, setCurrencyOn] = useState(false)
  const [currencyText, setCurrencyText] = useState("")

  // ---- object detection: stable detections are shown on screen and (optionally) spoken
  const announcer = useMemo(() => new ObjectAnnouncer(), [])
  const [stable, setStable] = useState<StableDetections | null>(null)
  const [voiceOn, setVoiceOn] = useState(true)
  const [viewSize, setViewSize] = useState({ width: 0, height: 0 })
  const lastSignature = useRef("")

  const onLayout = useCallback((e: LayoutChangeEvent) => {
    setViewSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })
  }, [])

  useEffect(() => {
    if (currencyOn) return
    const id = setInterval(() => {
      const snap = objectDetector.getStableDetections()
      const fresh = snap.ageMs >= 0 && snap.ageMs <= MAX_DETECTION_AGE_MS
      const shown = fresh ? snap : null
      // only re-render when something visible changed
      const signature = shown
        ? shown.detections.map(d => `${d.label}:${d.left.toFixed(2)},${d.top.toFixed(2)},${d.right.toFixed(2)},${d.bottom.toFixed(2)}`).join('|')
        : ""
      if (signature !== lastSignature.current) {
        lastSignature.current = signature
        setStable(shown)
      }
      // events must always be drained so the native queue cannot fill up, also when voice is off
      const phrase = announcer.handle(objectDetector.drainEvents())
      if (phrase && voiceOn) objectDetector.speak(phrase, false)
    }, 400)
    return () => clearInterval(id)
  }, [currencyOn, voiceOn, objectDetector, announcer])

  const toggleVoice = useCallback(() => {
    if (voiceOn) objectDetector.stopSpeaking()
    setVoiceOn(!voiceOn)
  }, [voiceOn, objectDetector])

  const toggleCurrency = useCallback(() => {
    const next = !currencyOn
    // either direction: silence object speech and forget stale detections/events so nothing old is announced later
    objectDetector.stopSpeaking()
    objectDetector.reset()
    announcer.reset()
    lastSignature.current = ""
    setStable(null)
    currencyRecognizer.setActive(next)
    // setActive can be refused (e.g. model failed to load): trust the native side
    setCurrencyOn(currencyRecognizer.isActive())
    setCurrencyText(next ? currencyRecognizer.getStatus() || currencyRecognizer.getError() : "")
  }, [currencyOn, currencyRecognizer, objectDetector, announcer])

  // show what the currency recogniser last said / any error
  useEffect(() => {
    if (!currencyOn) return
    const id = setInterval(() => {
      setCurrencyText(currencyRecognizer.getError() || currencyRecognizer.getStatus())
    }, 500)
    return () => clearInterval(id)
  }, [currencyOn, currencyRecognizer])

  // leaving the screen always switches currency mode (and its speech) off
  useEffect(() => () => currencyRecognizer.setActive(false), [currencyRecognizer])

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',

    enablePreviewSizedOutputBuffers: true,
    dropFramesWhileBusy: true,

    onFrame(frame) {
      'worklet';

      routeFrame(frame, objectDetector, currencyRecognizer)
      frame.dispose()
    }
  })

  if (!hasPermission) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.title}>Camera permission is required</Text>
        <Text style={styles.message}>VisionAid needs access to your camera to detect objects and obstacles.</Text>
      </View>
    )
  }

  if (device === undefined) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Camera not available</Text>
        <Text style={styles.message}>VisionAid could not find a camera on this device</Text>
      </View>
    )
  }

  return (
    <View style={styles.container} onLayout={onLayout}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[frameOutput]}
      />
      {!currencyOn && stable && stable.detections.map(d => {
        const box = boxToView(d, stable.frameWidth, stable.frameHeight, viewSize.width, viewSize.height)
        if (!box) return null
        return (
          <View
            key={d.classId}
            pointerEvents="none"
            importantForAccessibility="no-hide-descendants"
            style={[styles.box, { left: box.x, top: box.y, width: box.width, height: box.height }]}
          >
            <Text style={styles.boxLabel}>{d.label} {Math.round(d.score * 100)}%</Text>
          </View>
        )
      })}
      {!currencyOn && (
        <View style={styles.objectPanel} accessible accessibilityLabel={stable && stable.detections.length > 0 ? `Seen: ${stable.detections.map(d => describe(d)).join(', ')}` : "Nothing recognised yet"}>
          <Text style={styles.objectText}>
            {stable && stable.detections.length > 0 ? stable.detections.map(d => describe(d)).join('\n') : "Looking…"}
          </Text>
        </View>
      )}
      {!currencyOn && (
        <TouchableOpacity
          style={styles.voiceButton}
          onPress={toggleVoice}
          accessibilityRole="button"
          accessibilityLabel={voiceOn ? "Voice is on. Double tap to turn spoken object names off" : "Voice is off. Double tap to turn spoken object names on"}
        >
          <Text style={styles.voiceButtonText}>{voiceOn ? "Voice: on" : "Voice: off"}</Text>
        </TouchableOpacity>
      )}
      {currencyOn && (
        <View style={styles.currencyPanel} accessible accessibilityLabel={`Currency mode. ${currencyText}`}>
          <Text style={styles.currencyText}>{currencyText}</Text>
          <Text style={styles.currencyNote}>
            Names the banknote design only. It cannot tell if a note is genuine or still valid money.
          </Text>
        </View>
      )}
      <TouchableOpacity
        style={[styles.modeButton, currencyOn && styles.modeButtonOn]}
        onPress={toggleCurrency}
        accessibilityRole="button"
        accessibilityLabel={currencyOn ? "Currency mode is on. Double tap to go back to object detection" : "Currency mode is off. Double tap to recognise a banknote"}
      >
        <Text style={styles.modeButtonText}>{currencyOn ? "Stop currency mode" : "Currency mode"}</Text>
      </TouchableOpacity>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black'
  },

  permissionContainer: {
    flex: 1,
    backgroundColor: 'black',
    paddingHorizontal: 24,
    justifyContent: 'center',
    alignItems: 'center'
  },

  title: {
    color: 'white',
    fontSize: 26,
    fontWeight: 700,
    textAlign: 'center',
    marginBottom: 16,
    flexShrink: 1,
  },

  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00e676',
    borderRadius: 4
  },

  boxLabel: {
    alignSelf: 'flex-start',
    backgroundColor: '#00e676',
    color: 'black',
    fontSize: 12,
    fontWeight: 700,
    paddingHorizontal: 4
  },

  objectPanel: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 90,
    backgroundColor: 'rgba(0,0,0,0.65)',
    borderRadius: 12,
    padding: 12
  },

  objectText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 700
  },

  voiceButton: {
    position: 'absolute',
    top: 60,
    right: 16,
    minWidth: 64,
    minHeight: 64,
    borderRadius: 12,
    backgroundColor: '#37474f',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 6
  },

  voiceButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 700,
    textAlign: 'center'
  },

  currencyPanel: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.75)',
    borderRadius: 12,
    padding: 16
  },

  currencyText: {
    color: 'white',
    fontSize: 24,
    fontWeight: 700,
    textAlign: 'center'
  },

  currencyNote: {
    color: '#ddd',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8
  },

  modeButton: {
    position: 'absolute',
    bottom: 40,
    left: 24,
    right: 24,
    minHeight: 64,
    borderRadius: 16,
    backgroundColor: '#1565c0',
    justifyContent: 'center',
    alignItems: 'center'
  },

  modeButtonOn: {
    backgroundColor: '#b71c1c'
  },

  modeButtonText: {
    color: 'white',
    fontSize: 22,
    fontWeight: 700
  },

  message: {
    color: 'white',
    fontSize: 18,
    lineHeight: 27,
    textAlign: 'center',
    flexShrink: 1
  },
})

export default App;
