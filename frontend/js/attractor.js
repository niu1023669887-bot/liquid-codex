// Lorenz Attractor - Three.js background renderer
// σ=10, ρ=28, β=8/3 — classic parameters

(function () {
  const canvas = document.getElementById('attractor-canvas');
  if (!canvas) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 90);

  // Generate Lorenz attractor points
  const σ = 10, ρ = 28, β = 8 / 3;
  const dt = 0.005;
  const N = 60000;

  let x = 0.1, y = 0, z = 0;
  const positions = new Float32Array(N * 3);

  for (let i = 0; i < N; i++) {
    const dx = σ * (y - x);
    const dy = x * (ρ - z) - y;
    const dz = x * y - β * z;
    x += dx * dt;
    y += dy * dt;
    z += dz * dt;
    positions[i * 3]     = x * 1.2;
    positions[i * 3 + 1] = y * 1.2;
    positions[i * 3 + 2] = z * 1.2 - 30;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Gold color gradient along the attractor
  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const t = i / N;
    // Vary from dim gold to brighter gold
    colors[i * 3]     = 0.49 + t * 0.1;  // R
    colors[i * 3 + 1] = 0.42 + t * 0.08; // G
    colors[i * 3 + 2] = 0.10 + t * 0.05; // B
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Draw as line for smoother look
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.12,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const line = new THREE.Line(geometry, material);
  scene.add(line);

  // Also add a faint point cloud layer
  const pointMat = new THREE.PointsMaterial({
    color: 0xC9A962,
    size: 0.18,
    transparent: true,
    opacity: 0.06,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, pointMat);
  scene.add(points);

  // Resize
  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // Mouse / touch parallax state (lerp toward target each frame)
  let mouseOffsetX = 0, mouseOffsetY = 0;
  let lerpX = 0, lerpY = 0;

  function onPointerMove(cx, cy) {
    mouseOffsetX = (cx / window.innerWidth  - 0.5) * 0.45;
    mouseOffsetY = (cy / window.innerHeight - 0.5) * 0.22;
  }
  window.addEventListener('mousemove', e => onPointerMove(e.clientX, e.clientY));
  window.addEventListener('touchmove', e => {
    onPointerMove(e.touches[0].clientX, e.touches[0].clientY);
  }, { passive: true });
  window.addEventListener('mouseleave', () => { mouseOffsetX = 0; mouseOffsetY = 0; });

  // Animate — slow rotation + mouse parallax
  let frame = 0;
  function animate() {
    requestAnimationFrame(animate);
    frame++;
    // Smooth lerp toward mouse target (factor 0.04 = ~2s settle time)
    lerpX += (mouseOffsetX - lerpX) * 0.04;
    lerpY += (mouseOffsetY - lerpY) * 0.04;
    line.rotation.y = frame * 0.0003 + lerpX;
    line.rotation.x = Math.sin(frame * 0.0001) * 0.1 - lerpY;
    points.rotation.copy(line.rotation);
    renderer.render(scene, camera);
  }
  animate();
})();
