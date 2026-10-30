"use strict";

// Wood and felt colours in sRGB (0-1): base tone and the darker grain tone.
const WOODS = [
  { id: "oak", name: "Eiken", base: [0.78, 0.6, 0.4], grain: [0.58, 0.4, 0.24] },
  { id: "light", name: "Licht eiken", base: [0.87, 0.77, 0.63], grain: [0.7, 0.58, 0.44] },
  { id: "smoked", name: "Gerookt eiken", base: [0.5, 0.37, 0.26], grain: [0.33, 0.23, 0.15] },
  { id: "walnut", name: "Walnoot", base: [0.42, 0.28, 0.18], grain: [0.26, 0.16, 0.1] },
  { id: "black", name: "Zwart", base: [0.14, 0.13, 0.12], grain: [0.07, 0.065, 0.06] },
];
const FELTS = [
  { id: "black", name: "Zwart", color: [0.045, 0.045, 0.045] },
  { id: "grey", name: "Grijs", color: [0.22, 0.22, 0.23] },
];
const PANEL_WIDTH_M = 0.6;

const VERTEX = `#version 300 es
in vec2 aPos;
out vec2 vTex;
void main() {
  vTex = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
in vec2 vTex;
out vec4 outColor;

uniform sampler2D uPhoto;
uniform sampler2D uMask;
uniform sampler2D uLight;
uniform vec2 uSize;        // photo size in pixels
uniform mat3 uToWall;      // photo pixel -> wall uv (0..1)
uniform vec2 uWallM;       // wall size in metres
uniform vec2 uSpanM;       // covered part of the wall, left/right in metres
uniform float uTopM;       // top of the panels in metres from the floor
uniform float uSlat;       // slat width in metres
uniform float uGap;        // gap in metres
uniform vec3 uWoodBase;
uniform vec3 uWoodGrain;
uniform vec3 uFelt;
uniform float uExposure;
uniform float uSplit;      // 0..1, left of it shows the original photo

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

void main() {
  vec3 photo = texture(uPhoto, vTex).rgb;
  vec2 px = vTex * uSize;
  if (px.x < uSplit * uSize.x) { outColor = vec4(photo, 1.0); return; }

  vec3 q = uToWall * vec3(px, 1.0);
  vec2 uv = q.xy / q.z;
  float xm = uv.x * uWallM.x;            // metres from the left of the wall
  float ym = (1.0 - uv.y) * uWallM.y;    // metres from the floor
  float visible = texture(uMask, vTex).r;
  float onWall = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
  float inSpan = step(uSpanM.x, xm) * step(xm, uSpanM.y) * step(ym, uTopM);

  // Soft shadow the panels cast on the wallpaper next to them.
  float outside = max(max(uSpanM.x - xm, xm - uSpanM.y), ym - uTopM);
  float shadow = (1.0 - inSpan) * onWall * (1.0 - smoothstep(0.0, 0.05, outside)) * 0.35;
  vec3 base = photo * (1.0 - shadow * visible);

  // Slat pattern, starting at the left edge of the covered part.
  float x = xm - uSpanM.x;
  float pitch = uSlat + uGap;
  float id = floor(x / pitch);
  float f = x - id * pitch;
  float aa = fwidth(x) * 0.9;
  float slat = smoothstep(-aa, aa, f) * (1.0 - smoothstep(uSlat - aa, uSlat + aa, f));
  slat += smoothstep(-aa, aa, f - pitch);  // left edge of the next slat
  slat = clamp(slat, 0.0, 1.0);
  float s = clamp(f / uSlat, 0.0, 1.0);

  // Wood: grain running along the slat, a little colour variation per slat.
  float grain = noise(vec2(s * 3.0 + id * 13.1, ym * 5.0)) * 0.55
              + noise(vec2(s * 11.0 + id * 5.7, ym * 1.3 + id)) * 0.3
              + noise(vec2(s * 40.0 + id * 2.3, ym * 0.6)) * 0.15;
  vec3 wood = mix(uWoodBase, uWoodGrain, smoothstep(0.25, 0.85, grain));
  wood *= 0.93 + 0.14 * hash(vec2(id, 3.7));
  wood *= 1.0 + 0.08 * (1.0 - smoothstep(0.0, 0.15, s)) - 0.2 * smoothstep(0.78, 1.0, s);

  // Felt in the gaps, darker close to the slats.
  float g = clamp((f - uSlat) / max(uGap, 1e-4), 0.0, 1.0);
  vec3 felt = uFelt * (0.55 + 0.45 * sin(3.14159 * g)) * (0.9 + 0.2 * noise(px * 0.7));

  vec3 panel = mix(felt, wood, slat);
  vec3 light = texture(uLight, vTex).rgb * 2.0;
  panel *= light * uExposure;
  panel += (hash(px) - 0.5) * 0.02;  // a bit of photo grain

  float cover = visible * onWall * inSpan;
  outColor = vec4(mix(base, panel, cover), 1.0);
}`;

// Projective map from the unit square to a quad (Heckbert), as a row-major 3x3.
function squareToQuad(p) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = p;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / den;
  const h = (dx1 * dy3 - dx3 * dy1) / den;
  return [
    x1 - x0 + g * x1, x3 - x0 + h * x3, x0,
    y1 - y0 + g * y1, y3 - y0 + h * y3, y0,
    g, h, 1,
  ];
}

function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ];
}

function apply3(m, x, y) {
  const w = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Kan ${src} niet laden`));
    img.src = src;
  });
}

const $ = (id) => document.getElementById(id);

async function main() {
  const canvas = $("view");
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false });
  if (!gl) throw new Error("Je browser ondersteunt geen WebGL2.");

  const scene = await (await fetch("assets/scene.json", { cache: "no-cache" })).json();
  const [photo, mask, light, tidyPhoto, tidyMask] = await Promise.all(
    [scene.photo, scene.mask, scene.light, scene.tidyPhoto, scene.tidyMask].map(loadImage),
  );
  canvas.width = scene.width;
  canvas.height = scene.height;
  canvas.style.aspectRatio = `${scene.width} / ${scene.height}`;

  const program = createProgram(gl, VERTEX, FRAGMENT);
  gl.useProgram(program);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, "aPos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const texture = (img) => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  };
  // Units 0 and 1 hold the photo and mask, either as shot or with the toys painted out.
  const textures = {
    asShot: [texture(photo), texture(mask)],
    tidy: [texture(tidyPhoto), texture(tidyMask)],
  };
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, texture(light));
  const u = (name) => gl.getUniformLocation(program, name);
  gl.uniform1i(u("uPhoto"), 0);
  gl.uniform1i(u("uMask"), 1);
  gl.uniform1i(u("uLight"), 2);
  gl.uniform2f(u("uSize"), scene.width, scene.height);
  gl.uniform2f(u("uWallM"), scene.wallSizeM[0], scene.wallSizeM[1]);
  gl.viewport(0, 0, scene.width, scene.height);

  const state = {
    wood: WOODS[0],
    felt: FELTS[0],
    corners: scene.wall.map((p) => [...p]),
    split: 0.5,
  };

  buildSwatches($("woods"), WOODS, (w) => `rgb(${w.base.map((c) => c * 255).join(",")})`, (w) => {
    state.wood = w;
    render();
  });
  buildSwatches($("felts"), FELTS, (f) => `rgb(${f.color.map((c) => c * 255).join(",")})`, (f) => {
    state.felt = f;
    render();
  });

  function render() {
    textures[$("tidy").checked ? "tidy" : "asShot"].forEach((tex, unit) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
    });

    const toImage = squareToQuad(state.corners);
    const toWall = invert3(toImage);
    // GLSL mat3 is column-major.
    gl.uniformMatrix3fv(u("uToWall"), false, [
      toWall[0], toWall[3], toWall[6],
      toWall[1], toWall[4], toWall[7],
      toWall[2], toWall[5], toWall[8],
    ]);

    const [wallW, wallH] = scene.wallSizeM;
    const width = +$("width").value / 100;
    const height = +$("height").value / 100;
    // Centred on the TV; near the full width it is pushed back onto the wall.
    const tvU = apply3(toWall, scene.tvCenterX, (state.corners[0][1] + state.corners[3][1]) / 2)[0];
    const w = Math.min(width, wallW);
    const left = Math.min(Math.max(tvU * wallW - w / 2, 0), wallW - w);
    const span = [left, left + w];
    const top = Math.min(height, wallH);

    const slat = +$("slat").value / 1000;
    const gap = +$("gap").value / 1000;
    const exposure = +$("exposure").value / 100;
    gl.uniform2f(u("uSpanM"), span[0], span[1]);
    gl.uniform1f(u("uTopM"), top);
    gl.uniform1f(u("uSlat"), slat);
    gl.uniform1f(u("uGap"), gap);
    gl.uniform3fv(u("uWoodBase"), state.wood.base);
    gl.uniform3fv(u("uWoodGrain"), state.wood.grain);
    gl.uniform3fv(u("uFelt"), state.felt.color);
    gl.uniform1f(u("uExposure"), exposure);
    gl.uniform1f(u("uSplit"), $("compare").checked ? state.split : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const panels = Math.ceil(width / PANEL_WIDTH_M - 1e-6);
    $("widthOut").value = `${Math.round(width * 100)} cm (${panels} ${panels === 1 ? "paneel" : "panelen"} van 60 cm)`;
    $("heightOut").value = `${Math.round(height * 100)} cm${height >= wallH ? " (tot plafond)" : ""}`;
    $("slatOut").value = `${$("slat").value} mm`;
    $("gapOut").value = `${$("gap").value} mm`;
    $("exposureOut").value = `${$("exposure").value}%`;

    const split = $("split");
    split.hidden = !$("compare").checked || $("editCorners").checked;
    split.style.left = `${state.split * 100}%`;
    drawHandles();
  }

  // Before/after: drag anywhere on the photo.
  const stage = $("stage");
  let dragging = false;
  const setSplit = (e) => {
    const r = canvas.getBoundingClientRect();
    state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    render();
  };
  canvas.addEventListener("pointerdown", (e) => {
    if ($("editCorners").checked || !$("compare").checked) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    setSplit(e);
  });
  canvas.addEventListener("pointermove", (e) => dragging && setSplit(e));
  canvas.addEventListener("pointerup", () => (dragging = false));

  // Corner handles to fine-tune where the wall is.
  const svg = $("handles");
  svg.setAttribute("viewBox", `0 0 ${scene.width} ${scene.height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  function drawHandles() {
    const editing = $("editCorners").checked;
    $("resetCorners").hidden = !editing;
    stage.classList.toggle("editing", editing);
    if (!editing) return;
    svg.innerHTML =
      `<polygon points="${state.corners.map((p) => p.join(",")).join(" ")}"/>` +
      state.corners.map((p, i) => `<circle data-i="${i}" cx="${p[0]}" cy="${p[1]}" r="16"/>`).join("");
  }
  let corner = -1;
  svg.addEventListener("pointerdown", (e) => {
    if (e.target.tagName !== "circle") return;
    corner = +e.target.dataset.i;
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", (e) => {
    if (corner < 0) return;
    const r = svg.getBoundingClientRect();
    state.corners[corner] = [
      ((e.clientX - r.left) / r.width) * scene.width,
      ((e.clientY - r.top) / r.height) * scene.height,
    ];
    render();
  });
  svg.addEventListener("pointerup", () => (corner = -1));
  $("resetCorners").addEventListener("click", () => {
    state.corners = scene.wall.map((p) => [...p]);
    render();
  });

  $("controls").addEventListener("input", render);

  $("download").addEventListener("click", () => {
    const split = state.split;
    state.split = 0;
    render();
    canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `akupanelen-${state.wood.id}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
    state.split = split;
    render();
  });

  render();
}

function buildSwatches(container, items, css, onPick) {
  const group = container.id;
  items.forEach((item, i) => {
    const label = document.createElement("label");
    label.className = "swatch";
    label.innerHTML =
      `<input type="radio" name="${group}" value="${item.id}"${i === 0 ? " checked" : ""}>` +
      `<span class="chip" style="background:${css(item)}"></span><span class="name">${item.name}</span>`;
    label.querySelector("input").addEventListener("change", () => onPick(item));
    container.append(label);
  });
}

function createProgram(gl, vsSource, fsSource) {
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

main().catch((err) => {
  console.error(err);
  const el = $("error");
  el.textContent = err.message;
  el.hidden = false;
});
