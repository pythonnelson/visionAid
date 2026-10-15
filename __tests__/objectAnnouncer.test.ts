import { ObjectAnnouncer, directionOf, describe as say } from '../objectAnnouncer'
import { boxToView } from '../overlayLayout'

const obj = (label: string, classId: number, left: number, right: number, score = 0.8) =>
  ({ label, classId, score, left, right, top: 0.2, bottom: 0.8 })
const appeared = (o: any) => ({ type: 'appeared' as const, item: o })
const gone = (o: any) => ({ type: 'disappeared' as const, item: o })

test('direction comes from the horizontal centre of the box', () => {
  expect(directionOf({ left: 0.0, right: 0.4 })).toBe('left')
  expect(directionOf({ left: 0.4, right: 0.6 })).toBe('ahead')
  expect(directionOf({ left: 0.7, right: 1.0 })).toBe('right')
})

test('sentence names the object and where it is', () => {
  expect(say(obj('chair', 61, 0.0, 0.3))).toBe('chair on your left')
  expect(say(obj('person', 0, 0.4, 0.6))).toBe('person ahead')
})

test('nothing to say for no events or for objects that disappeared', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  expect(a.handle([])).toBeNull()
  expect(a.handle([gone(obj('chair', 61, 0, 0.3))])).toBeNull()
})

test('announces an appeared object once, capitalised', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  expect(a.handle([appeared(obj('chair', 61, 0.0, 0.3))])).toBe('Chair on your left')
})

test('several new objects become one sentence, important ones first, limited to 3', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  const text = a.handle([
    appeared(obj('chair', 61, 0.0, 0.3, 0.9)),
    appeared(obj('bottle', 43, 0.4, 0.6, 0.95)),
    appeared(obj('person', 0, 0.7, 1.0, 0.7)),
    appeared(obj('laptop', 72, 0.4, 0.6, 0.99)),
  ])
  expect(text).toBe('Person on your right, laptop ahead, bottle ahead')
})

test('the same object is not repeated within the cooldown, but is spoken again after it', () => {
  let t = 0
  const a = new ObjectAnnouncer({ now: () => t, cooldownMs: 10000 })
  const chair = obj('chair', 61, 0.0, 0.3)
  expect(a.handle([appeared(chair)])).not.toBeNull()
  t = 3000
  expect(a.handle([gone(chair)])).toBeNull()
  expect(a.handle([appeared(chair)])).toBeNull()           // flicker: back within the cooldown
  t = 11000
  expect(a.handle([appeared(chair)])).toBe('Chair on your left')
})

test('a different object is not blocked by the cooldown of another', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  a.handle([appeared(obj('chair', 61, 0.0, 0.3))])
  expect(a.handle([appeared(obj('person', 0, 0.4, 0.6))])).toBe('Person ahead')
})

test('reset forgets what was said', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  a.handle([appeared(obj('chair', 61, 0.0, 0.3))])
  a.reset()
  expect(a.handle([appeared(obj('chair', 61, 0.0, 0.3))])).toBe('Chair on your left')
})

test('keeps the most confident detection when one class appears twice in a batch', () => {
  const a = new ObjectAnnouncer({ now: () => 0 })
  expect(a.handle([appeared(obj('chair', 61, 0.0, 0.3, 0.6)), appeared(obj('chair', 61, 0.7, 1.0, 0.9))])).toBe('Chair on your right')
})

describe('boxToView (preview resizeMode cover)', () => {
  test('frame with the same aspect as the view maps linearly', () => {
    expect(boxToView({ left: 0.25, top: 0.5, right: 0.75, bottom: 1 }, 720, 1280, 360, 640)).toEqual({ x: 90, y: 320, width: 180, height: 320 })
  })
  test('taller view crops the sides of the frame (cover)', () => {
    // frame 720x1280 in a 1080x2400 view: scale = 2400/1280 = 1.875 -> width 1350, 135 px cropped left and right
    const b = boxToView({ left: 0, top: 0, right: 1, bottom: 1 }, 720, 1280, 1080, 2400)!
    expect(b.x).toBe(0); expect(b.width).toBe(1080); expect(b.height).toBe(2400)
    const mid = boxToView({ left: 0.4, top: 0, right: 0.6, bottom: 1 }, 720, 1280, 1080, 2400)!
    expect(mid.x).toBeCloseTo(0.4 * 1350 - 135); expect(mid.width).toBeCloseTo(270)
  })
  test('unknown sizes or boxes outside the visible area give null', () => {
    expect(boxToView({ left: 0, top: 0, right: 1, bottom: 1 }, 0, 0, 100, 100)).toBeNull()
    expect(boxToView({ left: 0, top: 0, right: 0.05, bottom: 1 }, 720, 1280, 1080, 2400)).toBeNull()   // fully in the cropped margin
  })
})
