import { useCallback, useEffect, useMemo, useState } from "react"
import { StyleSheet, Text, TouchableOpacity, View } from "react-native"
import { Camera, useCameraDevice, useCameraPermission, useFrameOutput } from "react-native-vision-camera"
import { NitroModules } from "react-native-nitro-modules"
import type { VisionAidObjectDetector, VisionAidCurrencyRecognizer } from "./VisionAidObjectDetector/src"
import { routeFrame } from "./frameRouter"

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

  const toggleCurrency = useCallback(() => {
    const next = !currencyOn
    currencyRecognizer.setActive(next)
    // setActive can be refused (e.g. model failed to load): trust the native side
    setCurrencyOn(currencyRecognizer.isActive())
    setCurrencyText(next ? currencyRecognizer.getStatus() || currencyRecognizer.getError() : "")
  }, [currencyOn, currencyRecognizer])

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
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[frameOutput]}
      />
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
