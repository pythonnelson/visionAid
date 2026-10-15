import { routeFrame } from '../frameRouter'

const frame = {} as any

function mocks(active: boolean) {
  return {
    detector: { detect: jest.fn() },
    currency: { isActive: jest.fn(() => active), analyze: jest.fn() },
  }
}

test('object detector gets the frame while currency mode is off', () => {
  const { detector, currency } = mocks(false)
  expect(routeFrame(frame, detector, currency)).toBe('objects')
  expect(detector.detect).toHaveBeenCalledWith(frame)
  expect(currency.analyze).not.toHaveBeenCalled()
})

test('currency recogniser gets the frame while currency mode is on, and the detector does not', () => {
  const { detector, currency } = mocks(true)
  expect(routeFrame(frame, detector, currency)).toBe('currency')
  expect(currency.analyze).toHaveBeenCalledWith(frame)
  expect(detector.detect).not.toHaveBeenCalled()
})

test('switching back returns frames to the object detector', () => {
  let active = true
  const detector = { detect: jest.fn() }
  const currency = { isActive: () => active, analyze: jest.fn() }
  routeFrame(frame, detector, currency)
  active = false
  routeFrame(frame, detector, currency)
  expect(currency.analyze).toHaveBeenCalledTimes(1)
  expect(detector.detect).toHaveBeenCalledTimes(1)
})
