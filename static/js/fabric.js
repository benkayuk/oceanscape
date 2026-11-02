// The "Concept" style (Display > Style): the road as a sheet of perforated fabric, after Territory Studio's
// original ADAS concepts for the Ocean (territorystudio.com/project/fiskerocean). Lanes are soft ridges pushed
// up through the sheet with the lane lines glowing along them, the cars around ours emerge from it as draped
// shapes, and the car we follow sends ripples across the fabric in front of us, green to red with the gap.
//
// One mesh in the car frame, dense near the car and coarser down the road, displaced in its vertex shader by the
// road field (ground.js): the ridges sit at the lane boundaries in the field's own coordinates (offset across the
// reference arc or polyline), so they bend with the road and grow and fade with it as the classic ground does.
// The perforations are fixed to the ground (through uGroundInv), so the sheet moves under the car.
//
// Scene frame: X right, Y up, Z backward; the car frame (x ahead of the front bumper, y left) is x = -Z, y = -X.
import * as THREE from '../vendor/three.module.min.js';
import { FIELD_GLSL } from './ground.js';

export const RIDGE_MAX = 12;
export const OBJ_MAX = 8;
const RIDGE_H = 0.3;           // m: a lane line's ridge at weight 1
const OBJ_BUMP = 0.42;         // m: how far a car's footprint lifts the sheet (its lower part is inside it)
const PITCH = 0.085;           // m between perforations
const DASH_P = 12, DASH_ON = 3;   // m: dash period and length (the classic dashes are 3 + 9)

// the sheet's vertices: rows down the road (fine near the car, coarser far), columns across (fine over the road)
function sheetGeometry() {
  const xs = [], ys = [];
  for (let x = -45; x < 165;) { xs.push(x); x += x < 30 ? 0.3 : 0.3 + (x - 30) * 0.014; }
  for (let y = 0; y < 160;) { ys.push(y); y += y < 12 ? 0.2 : 0.2 + (y - 12) * 0.025; }
  const yy = [...ys.slice(1).map(v => -v).reverse(), ...ys];
  const nx = xs.length, ny = yy.length;
  const pos = new Float32Array(nx * ny * 3);
  let k = 0;
  for (const x of xs) for (const y of yy) { pos[k++] = -y; pos[k++] = 0; pos[k++] = -x; }
  const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
  k = 0;
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const a = i * ny + j, b = a + 1, c = a + ny, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

const VERT = `
#define RIDGE_MAX ${RIDGE_MAX}
#define OBJ_MAX ${OBJ_MAX}
uniform vec2 uRidgeD[RIDGE_MAX];   // each ridge: offset across the road (m, the field's coordinates), weight
uniform float uRidgeS[RIDGE_MAX];  // ...its half-width (m)
uniform int uRidgeN;
uniform vec4 uObj[OBJ_MAX];        // each car: center (car frame), heading (cos, sin)
uniform vec4 uObjS[OBJ_MAX];       // ...half length, half width, lift (0..1), unused
uniform int uObjN;
varying vec3 vWorld;
varying vec2 vCar;
varying vec2 vSD;
varying float vMask;
varying float vLift;
${FIELD_GLSL}
void main() {
  vec3 p = position;
  vec2 car = vec2(-p.z, -p.x);
  vec2 sd = roadSD(car);
  float rd = roadDist(sd);
  float mask = smoothstep(0.0, 1.0, -rd);
  float h = 0.0;
  for (int i = 0; i < RIDGE_MAX; i++) {
    if (i >= uRidgeN) break;
    float d = (sd.y - uRidgeD[i].x) / uRidgeS[i];
    h += uRidgeD[i].y * exp(-d * d);
  }
  h *= ${RIDGE_H.toFixed(3)} * mask;
  float lift = 0.0;
  for (int i = 0; i < OBJ_MAX; i++) {
    if (i >= uObjN) break;
    vec2 q = car - uObj[i].xy;
    q = vec2(dot(q, uObj[i].zw), q.y * uObj[i].z - q.x * uObj[i].w);
    vec2 e = abs(q) - uObjS[i].xy;
    float od = length(max(e, 0.0)) + min(max(e.x, e.y), 0.0);
    lift = max(lift, uObjS[i].z * (1.0 - smoothstep(-0.5, 1.8, od)));
  }
  p.y = h + lift * ${OBJ_BUMP.toFixed(3)} - 0.012;
  vec4 w = modelMatrix * vec4(p, 1.0);
  vWorld = w.xyz;
  vCar = car;
  vSD = sd;
  vMask = mask;
  vLift = lift;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const FRAG = `
#define RIDGE_MAX ${RIDGE_MAX}
uniform mat4 uGroundInv;
uniform vec2 uRidgeD[RIDGE_MAX];
uniform vec4 uRidgeC[RIDGE_MAX];   // each ridge's line: color (linear), dashed (1) or solid (0); black = no line
uniform int uRidgeN;
uniform vec3 uBase, uHole, uFog, uLight;
uniform vec2 uFogR;                // fog near, far (m from the camera)
uniform float uDark, uOdo, uTime;
uniform vec4 uLead;                // the car we follow: ripple center (car frame), strength, unused
uniform vec3 uLeadC;
varying vec3 vWorld;
varying vec2 vCar;
varying vec2 vSD;
varying float vMask;
varying float vLift;
void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  if (n.y < 0.0) n = -n;
  float diff = clamp(dot(n, uLight), 0.0, 1.0);
  // perforations fixed to the ground; averaged out where they'd be under a pixel apart (no moire down the road)
  vec2 g = (uGroundInv * vec4(vWorld, 1.0)).xz;
  vec2 cell = (fract(g / ${PITCH}) - 0.5) * ${PITCH};
  float fw = length(fwidth(g));
  float rad = ${(PITCH * 0.3).toFixed(4)};
  float dotm = 1.0 - smoothstep(rad - fw * 0.5, rad + fw * 0.5, length(cell));
  float vis = 1.0 - smoothstep(${(PITCH * 0.3).toFixed(4)}, ${(PITCH * 0.8).toFixed(4)}, fw);
  float area = ${(Math.PI * 0.09).toFixed(4)};
  float holes = mix(area, dotm, vis);
  // a low light from ahead and to the left brings the ridges out; a sheen where they face it
  vec3 view = normalize(cameraPosition - vWorld);
  float spec = pow(clamp(dot(n, normalize(uLight + view)), 0.0, 1.0), 24.0);
  float lit = 0.38 + 0.85 * diff;
  vec3 col = uBase * lit + spec * 0.18 * (1.0 - holes);
  col = mix(col, uHole * (0.6 + 0.4 * lit), holes * 0.85);
  // the lines: a core and a halo along each ridge, shining out of the perforations
  vec3 glow = vec3(0.0);
  float through = mix(0.45, dotm, vis);
  for (int i = 0; i < RIDGE_MAX; i++) {
    if (i >= uRidgeN) break;
    vec4 c = uRidgeC[i];
    if (c.r + c.g + c.b < 1e-4) continue;
    float d = vSD.y - uRidgeD[i].x;
    float core = exp(-d * d / 0.006), halo = exp(-d * d / 0.16);
    float on = 1.0;
    if (c.a > 0.5) {
      float ph = fract((vSD.x + uOdo) / ${DASH_P.toFixed(1)});
      on = smoothstep(0.0, 0.015, ph) * (1.0 - smoothstep(${(DASH_ON / DASH_P - 0.02).toFixed(3)}, ${(DASH_ON / DASH_P).toFixed(3)}, ph));
    }
    glow += c.rgb * on * (core * (0.55 + 0.45 * through) + halo * 0.35 * through);
  }
  glow *= vMask;
  // ripples running out from in front of the car we follow
  vec2 lp = vCar - uLead.xy;
  lp.x *= 1.35;
  float lr = length(lp);
  float wave = pow(0.5 + 0.5 * cos(lr * 6.2832 / 0.85 - uTime * 4.0), 5.0);
  float env = smoothstep(0.25, 0.8, lr) * (1.0 - smoothstep(1.4, 3.0, lr));
  glow += uLeadC * uLead.z * env * wave * (0.35 + 0.65 * through) * (uDark > 0.5 ? 1.0 : 1.6);
  float gi = max(glow.r, max(glow.g, glow.b));
  if (uDark > 0.5) col += glow * 1.6;
  else col = mix(col, glow / max(gi, 1e-4), min(1.0, gi) * 0.92);
  // into the background: fog with distance, and the sheet's own sides
  float fd = length(vWorld - cameraPosition);
  float fog = smoothstep(uFogR.x, uFogR.y, fd);
  col = mix(col, uFog, fog);
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`;

const LOOK = {
  light: { base: 0xdcdcdc, hole: 0x8a8a8a, top: 0xf2f2f2, horizon: 0xd6d6d6, object: 0xb9b9b9, lead: 0x9a9a9a },
  dark: { base: 0x555555, hole: 0x1c1c1c, top: 0x1a1a1a, horizon: 0x3a3a3a, object: 0x3a3a3a, lead: 0x474747 },
};
export const CONCEPT_COLORS = {
  amber: new THREE.Color(0xffa62b), blue: new THREE.Color(0x3d8bff), red: new THREE.Color(0xff3b30),
  green: new THREE.Color(0x3ddc6a), yellow: new THREE.Color(0xffd23a), orange: new THREE.Color(0xff8a1f),
};

export class Fabric {
  constructor(field) {
    const U = {
      uRidgeD: { value: Array.from({ length: RIDGE_MAX }, () => new THREE.Vector2()) },
      uRidgeS: { value: new Float32Array(RIDGE_MAX).fill(0.5) },
      uRidgeC: { value: Array.from({ length: RIDGE_MAX }, () => new THREE.Vector4()) },
      uRidgeN: { value: 0 },
      uObj: { value: Array.from({ length: OBJ_MAX }, () => new THREE.Vector4()) },
      uObjS: { value: Array.from({ length: OBJ_MAX }, () => new THREE.Vector4()) },
      uObjN: { value: 0 },
      uBase: { value: new THREE.Color() }, uHole: { value: new THREE.Color() }, uFog: { value: new THREE.Color() },
      uLight: { value: new THREE.Vector3(-0.55, 0.55, -0.65).normalize() },
      uFogR: { value: new THREE.Vector2(60, 170) },
      uDark: { value: 0 }, uOdo: { value: 0 }, uTime: { value: 0 },
      uLead: { value: new THREE.Vector4() }, uLeadC: { value: new THREE.Color() },
    };
    this.uniforms = U;
    const mat = new THREE.ShaderMaterial({ uniforms: Object.assign({}, field.uniforms, U), vertexShader: VERT, fragmentShader: FRAG });
    this.mesh = new THREE.Mesh(sheetGeometry(), mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1;
    this.mesh.visible = false;
    // the glowing bar across the rear of each car ahead
    const barMat = new THREE.MeshBasicMaterial({ color: 0xff2a1a, transparent: true, depthWrite: false, toneMapped: false });
    this.bars = Array.from({ length: OBJ_MAX }, () => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1, 0.07, 0.04), barMat.clone());
      m.visible = false;
      return m;
    });
    this.group = new THREE.Group();
    this.group.add(this.mesh, ...this.bars);
    this.lead = { s: 0, color: new THREE.Color(CONCEPT_COLORS.green) };
    this.look = LOOK.light;
  }

  /** The colors for day or night; returns the look (for the scene's background and objects). */
  setTheme(dark) {
    const L = this.look = dark ? LOOK.dark : LOOK.light, U = this.uniforms;
    U.uBase.value.set(L.base);
    U.uHole.value.set(L.hole);
    U.uFog.value.set(L.horizon);
    U.uDark.value = dark ? 1 : 0;
    return L;
  }

  /** A vertical gradient for the scene's background: lighter (day) or darker (night) overhead, the horizon at the fog's color. */
  background() {
    const c = document.createElement('canvas');
    c.width = 4; c.height = 256;
    const ctx = c.getContext('2d');
    const gr = ctx.createLinearGradient(0, 0, 0, 256);
    const hex = (v) => '#' + v.toString(16).padStart(6, '0');
    gr.addColorStop(0, hex(this.look.top));
    gr.addColorStop(0.62, hex(this.look.horizon));
    gr.addColorStop(1, hex(this.look.horizon));
    ctx.fillStyle = gr;
    ctx.fillRect(0, 0, 4, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  /**
   * This frame: ridges [{d, w, s, color: THREE.Color | null, dashed}] (d: m across the road field's reference, w:
   * weight, s: half-width m), cars [{x, y, cos, sin, hl, hw, alpha, ahead, lead, gapS}] (car frame centers),
   * the odometer (m, for the dashes) and the clock.
   */
  update(ridges, cars, odo, clock, dt) {
    const U = this.uniforms;
    const n = Math.min(RIDGE_MAX, ridges.length);
    for (let i = 0; i < n; i++) {
      const r = ridges[i];
      U.uRidgeD.value[i].set(r.d, r.w);
      U.uRidgeS.value[i] = r.s;
      const c = r.color;
      U.uRidgeC.value[i].set(c ? c.r : 0, c ? c.g : 0, c ? c.b : 0, r.dashed ? 1 : 0);
    }
    U.uRidgeN.value = n;
    const m = Math.min(OBJ_MAX, cars.length);
    let lead = null;
    for (let i = 0; i < m; i++) {
      const o = cars[i];
      U.uObj.value[i].set(o.x, o.y, o.cos, o.sin);
      U.uObjS.value[i].set(o.hl, o.hw, o.alpha, 0);
      if (o.lead && (!lead || o.x < lead.x)) lead = o;
      // the rear bar: across the back of the car, at lamp height, facing us
      const b = this.bars[i];
      b.visible = o.ahead && o.alpha > 0.05;
      if (b.visible) {
        const rx = o.x - o.cos * o.hl, ry = o.y - o.sin * o.hl;
        b.position.set(-ry + o.sin * 0.03, 0.78 * o.hgt, -rx + o.cos * 0.03);
        b.rotation.y = Math.atan2(o.sin, o.cos);
        b.scale.set(o.hw * 1.5, 1, 1);
        b.material.opacity = o.alpha * (o.braking ? 1 : 0.75);
      }
    }
    for (let i = m; i < OBJ_MAX; i++) this.bars[i].visible = false;
    U.uObjN.value = m;
    // the ripples: green with a long gap, through yellow and orange, red when close; eased so they don't blink
    const L = this.lead;
    const target = lead ? lead.alpha : 0;
    L.s += (target - L.s) * Math.min(1, dt * 4);
    if (lead) {
      const g = lead.gapS;
      const want = g > 2.2 ? CONCEPT_COLORS.green : g > 1.5 ? CONCEPT_COLORS.yellow : g > 1.0 ? CONCEPT_COLORS.orange : CONCEPT_COLORS.red;
      L.color.lerp(want, Math.min(1, dt * 4));
      const back = lead.hl + Math.min(4, Math.max(2.5, 0.5 * (lead.x - lead.hl)));   // between us and its rear bumper
      const rx = lead.x - lead.cos * back, ry = lead.y - lead.sin * back;
      U.uLead.value.set(rx, ry, L.s, 0);
    } else U.uLead.value.z = L.s;
    U.uLeadC.value.copy(L.color);
    U.uOdo.value = odo % DASH_P;
    U.uTime.value = clock % 1000;
  }
}
