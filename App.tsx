import { useEffect } from "react"
import { StyleSheet, Text, View } from "react-native"
import { Camera, useCameraDevice, useCameraPermission } from "react-native-vision-camera"

function App(): React.JSX.Element {
  const device = useCameraDevice("back")
  const { hasPermission, requestPermission } = useCameraPermission()

  useEffect(() => {
    if (!hasPermission) {
      requestPermission()
    }
  }, [hasPermission, requestPermission])

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
      />
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

  message: {
    color: 'white',
    fontSize: 18,
    lineHeight: 27,
    textAlign: 'center',
    flexShrink: 1
  },
})

export default App;
