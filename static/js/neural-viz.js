/**
 * NEUROSCAN — 3D Neural Network Visualization
 * Three.js scene with InstancedMesh neurons, bloom post-processing,
 * and layer-by-layer progressive rendering.
 *
 * Color semantics:
 *   Polarity (hue)  — Cyan=inhibitory, Purple=balanced, Orange=excitatory
 *   Magnitude (size) — larger sphere = stronger activation
 *   Variance (ring)  — high-variance neurons pulse gently
 *   Layer depth       — subtle warm shift in later layers
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ── Constants ────────────────────────────────────────────────
const NEURON_BASE_SIZE = 0.08;
const NEURON_MAX_SIZE = 0.4;
const MAX_NEURONS_PER_LAYER = 100;
const CONNECTION_OPACITY = 0.12;
const BLOOM_STRENGTH = 1.4;
const BLOOM_RADIUS = 0.5;
const BLOOM_THRESHOLD = 0.2;

// ── Brain layout constants ──────────────────────────────────
// Neurons are arranged on concentric ellipsoidal shells that form
// a brain shape. Early layers (input/cortex) are on the outer shell,
// later layers (deep processing) nest inward toward the core.
const BRAIN_RADIUS_X = 10;          // width (left-right, widest)
const BRAIN_RADIUS_Y = 7.5;         // height (top-bottom)
const BRAIN_RADIUS_Z = 9;           // depth (front-back)
const BRAIN_INNER_SCALE = 0.22;     // innermost shell as fraction of outer
const BRAIN_MIDLINE_GAP = 0.7;      // gap between hemispheres (longitudinal fissure)
const BRAIN_JITTER = 0.2;           // organic randomness
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

// ── Polarity color palette ──────────────────────────────────
// Neurons are colored by their signed activation pattern:
//   polarity = mean(per_token) / mean(|per_token|)
//   -1 = purely inhibitory, 0 = balanced, +1 = purely excitatory
const COLOR_INHIBIT_DEEP = new THREE.Color(0x005577);   // deep teal (strong -)
const COLOR_INHIBIT      = new THREE.Color(0x00e5ff);   // cyan (inhibitory)
const COLOR_BALANCED     = new THREE.Color(0xb44aff);   // purple (balanced)
const COLOR_EXCITE       = new THREE.Color(0xffaa00);   // amber (excitatory)
const COLOR_EXCITE_HOT   = new THREE.Color(0xff3333);   // red-hot (strong +)
const COLOR_WHITE        = new THREE.Color(0xffffff);   // extreme magnitude

// Layer depth tint: earlier layers are cooler, later layers are warmer
const LAYER_TINT_EARLY = new THREE.Color(0x6688cc);     // cool blue tinge
const LAYER_TINT_LATE  = new THREE.Color(0xcc8866);     // warm amber tinge
const LAYER_TINT_STRENGTH = 0.12; // how much layer tint affects final color

const _tmpMatrix = new THREE.Matrix4();
const _tmpColor = new THREE.Color();
const _tmpVec = new THREE.Vector3();
const _tmpTint = new THREE.Color();

/**
 * Compute neuron color from per-token activations.
 *
 * polarity: fraction of signed mean vs absolute mean [-1, +1]
 * magnitude: normalized 0..1 activation strength
 * layerFrac: 0..1 position in the network (early..late)
 */
function neuronColor(polarity, magnitude, layerFrac) {
    // Map polarity to base hue
    // polarity: -1 → cyan, 0 → purple, +1 → amber
    let base;
    if (polarity < -0.5) {
        // Deep inhibitory → cyan
        base = _tmpColor.lerpColors(COLOR_INHIBIT_DEEP, COLOR_INHIBIT, (polarity + 1) * 2);
    } else if (polarity < 0) {
        // Mild inhibitory → purple
        base = _tmpColor.lerpColors(COLOR_INHIBIT, COLOR_BALANCED, (polarity + 0.5) * 2);
    } else if (polarity < 0.5) {
        // Mild excitatory → amber
        base = _tmpColor.lerpColors(COLOR_BALANCED, COLOR_EXCITE, polarity * 2);
    } else {
        // Strong excitatory → red-hot
        base = _tmpColor.lerpColors(COLOR_EXCITE, COLOR_EXCITE_HOT, (polarity - 0.5) * 2);
    }

    // Brighten toward white for very high magnitude
    if (magnitude > 0.7) {
        const whiteMix = (magnitude - 0.7) / 0.3;  // 0..1
        base.lerp(COLOR_WHITE, whiteMix * 0.5);
    }

    // Apply layer depth tint
    _tmpTint.lerpColors(LAYER_TINT_EARLY, LAYER_TINT_LATE, layerFrac);
    base.lerp(_tmpTint, LAYER_TINT_STRENGTH);

    return base.clone();
}

/**
 * Compute statistics from per-token activation array.
 */
function neuronStats(perToken) {
    if (!perToken || perToken.length === 0) {
        return { polarity: 0, variance: 0, absMean: 0 };
    }
    let sum = 0, absSum = 0, sqSum = 0;
    for (let i = 0; i < perToken.length; i++) {
        sum += perToken[i];
        absSum += Math.abs(perToken[i]);
        sqSum += perToken[i] * perToken[i];
    }
    const n = perToken.length;
    const mean = sum / n;
    const absMean = absSum / n;
    const variance = sqSum / n - mean * mean;

    // polarity: -1 (all negative) to +1 (all positive), 0 = balanced
    const polarity = absMean > 0.001 ? mean / absMean : 0;
    // normalized variance (coefficient of variation)
    const cv = absMean > 0.001 ? Math.sqrt(Math.max(0, variance)) / absMean : 0;

    return { polarity, variance: cv, absMean };
}


class NeuralViz {
    constructor() {
        this._container = null;
        this._scene = null;
        this._camera = null;
        this._renderer = null;
        this._composer = null;
        this._controls = null;
        this._clock = new THREE.Clock();

        // Neuron data
        this._layerMeshes = [];        // InstancedMesh per layer
        this._connectionLines = [];     // Line objects between layers
        this._layerLabels = [];         // CSS labels
        this._neuronData = [];          // raw data per layer
        this._neuronColors = [];        // THREE.Color per [layer][instance]
        this._neuronVariance = [];      // variance per [layer][instance]
        this._neuronNorms = [];         // normalized activation [0-1] per [layer][instance]
        this._neuronScales = [];        // base scale per [layer][instance]
        this._nLayers = 0;
        this._totalLayers = 12;        // updated on scan complete
        this._tokens = [];

        // Interaction
        this._raycaster = new THREE.Raycaster();
        this._mouse = new THREE.Vector2();
        this._hoveredNeuron = null;
        this._showConnections = true;
        this._showLabels = false;
        this._threshold = 0.1;
        this._prevThreshold = 0.1;   // track changes to avoid per-frame matrix rewrites

        // Layer label overlays (HTML divs projected from 3D)
        this._labelContainer = null;
        this._labelElements = [];  // DOM elements for each layer

        // Animation
        this._animating = false;
        this._targetLayerAlpha = [];
        this._currentLayerAlpha = [];
        this._time = 0;
    }

    init(container) {
        this._container = container;
        const w = container.clientWidth;
        const h = container.clientHeight;

        // Scene
        this._scene = new THREE.Scene();
        this._scene.fog = new THREE.FogExp2(0x0a0a0f, 0.015);

        // Camera — positioned to view the brain from above-front
        this._camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 500);
        this._camera.position.set(0, 14, 24);
        this._camera.lookAt(0, 0, 0);

        // Renderer
        this._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this._renderer.setSize(w, h);
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this._renderer.toneMappingExposure = 1.0;
        container.appendChild(this._renderer.domElement);

        // Layer label overlay container (sits on top of the canvas)
        this._labelContainer = document.createElement('div');
        this._labelContainer.className = 'layer-label-container';
        container.appendChild(this._labelContainer);

        // Post-processing: bloom
        this._composer = new EffectComposer(this._renderer);
        this._composer.addPass(new RenderPass(this._scene, this._camera));
        const bloom = new UnrealBloomPass(
            new THREE.Vector2(w, h),
            BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD
        );
        this._composer.addPass(bloom);

        // Controls — orbit around brain center
        this._controls = new OrbitControls(this._camera, this._renderer.domElement);
        this._controls.enableDamping = true;
        this._controls.dampingFactor = 0.08;
        this._controls.target.set(0, 0, 0);
        this._controls.minDistance = 5;
        this._controls.maxDistance = 80;

        // Lighting
        this._scene.add(new THREE.AmbientLight(0x221133, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.2);
        dirLight.position.set(10, 30, 10);
        this._scene.add(dirLight);

        // Ground grid
        this._addGroundGrid();

        // Resize
        this._onResize = () => {
            const w2 = container.clientWidth;
            const h2 = container.clientHeight;
            this._camera.aspect = w2 / h2;
            this._camera.updateProjectionMatrix();
            this._renderer.setSize(w2, h2);
            this._composer.setSize(w2, h2);
        };
        window.addEventListener('resize', this._onResize);

        // Mouse for raycasting
        this._renderer.domElement.addEventListener('mousemove', (e) => {
            const rect = this._renderer.domElement.getBoundingClientRect();
            this._mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this._mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        });

        // Click to select neuron
        this._onNeuronClick = null;
        this._renderer.domElement.addEventListener('click', () => {
            if (this._hoveredNeuron && this._onNeuronClick) {
                this._onNeuronClick(this._hoveredNeuron);
            }
        });
        this._renderer.domElement.style.cursor = 'default';

        // Start render loop
        this._animating = true;
        this._animate();
    }

    _addGroundGrid() {
        // Subtle concentric rings beneath the brain (like a scan platform)
        const gridGeo = new THREE.BufferGeometry();
        const verts = [];
        const ringCount = 6;
        const maxRadius = 14;
        const segments = 64;
        const groundY = -(BRAIN_RADIUS_Y + 1.5);
        for (let r = 0; r < ringCount; r++) {
            const radius = maxRadius * (r + 1) / ringCount;
            for (let s = 0; s < segments; s++) {
                const a1 = (s / segments) * Math.PI * 2;
                const a2 = ((s + 1) / segments) * Math.PI * 2;
                verts.push(Math.cos(a1) * radius, groundY, Math.sin(a1) * radius);
                verts.push(Math.cos(a2) * radius, groundY, Math.sin(a2) * radius);
            }
        }
        // Cross-hairs
        for (let a = 0; a < 4; a++) {
            const angle = (a / 4) * Math.PI * 2;
            verts.push(0, groundY, 0);
            verts.push(Math.cos(angle) * maxRadius, groundY, Math.sin(angle) * maxRadius);
        }
        gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
        const gridMat = new THREE.LineBasicMaterial({ color: 0xb44aff, opacity: 0.05, transparent: true });
        this._scene.add(new THREE.LineSegments(gridGeo, gridMat));
    }

    clear() {
        for (const mesh of this._layerMeshes) {
            if (!mesh) continue;
            this._scene.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
        }
        this._layerMeshes = [];
        for (const line of this._connectionLines) {
            this._scene.remove(line);
            line.geometry.dispose();
            line.material.dispose();
        }
        this._connectionLines = [];
        this._neuronData = [];
        this._neuronColors = [];
        this._neuronVariance = [];
        this._targetLayerAlpha = [];
        this._currentLayerAlpha = [];

        // Remove layer label DOM elements
        for (const el of this._labelElements) el.remove();
        this._labelElements = [];
    }

    /**
     * Add a single layer of neuron data (called progressively).
     */
    addLayer(layerData) {
        const layerIdx = layerData.layer;
        const neurons = layerData.neurons || [];

        this._neuronData[layerIdx] = neurons;
        const layerFrac = this._totalLayers > 1 ? layerIdx / (this._totalLayers - 1) : 0;

        // Shell scale: outer (layer 0 = cortex) → inner (last layer = core)
        const shellScale = 1.0 - layerFrac * (1.0 - BRAIN_INNER_SCALE);

        // Brain ellipsoid radii for this layer's shell
        const rx = BRAIN_RADIUS_X * shellScale;
        const ry = BRAIN_RADIUS_Y * shellScale;
        const rz = BRAIN_RADIUS_Z * shellScale;

        const maxAct = Math.max(...neurons.map(n => n.mean_activation), 0.001);
        const count = Math.min(neurons.length, MAX_NEURONS_PER_LAYER);

        // Compute per-neuron stats from signed per_token data
        const stats = [];
        for (let i = 0; i < count; i++) {
            stats.push(neuronStats(neurons[i].per_token));
        }

        // Create mesh — MeshBasicMaterial so per-instance colors render
        // directly (not multiplied against a shared emissive). Bloom
        // post-processing creates the neon glow from the raw color.
        const geo = new THREE.SphereGeometry(1, 14, 10);
        const mat = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0,
        });
        const mesh = new THREE.InstancedMesh(geo, mat, count);
        mesh.userData = { layerIdx, neurons: neurons.slice(0, count), maxAct, stats };

        // Store colors, variance, norms, and scales for animation + threshold
        const layerColors = [];
        const layerVariance = [];
        const layerNorms = [];
        const layerScales = [];

        // Seed RNG per layer for deterministic jitter (same positions on re-render)
        let rngSeed = layerIdx * 7919 + 1;
        const seededRandom = () => {
            rngSeed = (rngSeed * 16807 + 0) % 2147483647;
            return rngSeed / 2147483647;
        };

        for (let i = 0; i < count; i++) {
            const norm = neurons[i].mean_activation / maxAct;
            const scale = NEURON_BASE_SIZE + norm * (NEURON_MAX_SIZE - NEURON_BASE_SIZE);

            // Fibonacci sphere: evenly distributed points on unit sphere
            const t = count > 1 ? i / (count - 1) : 0.5;
            const yNorm = 1 - t * 2; // +1 (top) to -1 (bottom)
            const radiusAtY = Math.sqrt(Math.max(0, 1 - yNorm * yNorm));
            const theta = GOLDEN_ANGLE * i;

            let px = Math.cos(theta) * radiusAtY * rx;
            let py = yNorm * ry;
            let pz = Math.sin(theta) * radiusAtY * rz;

            // Midline gap — longitudinal fissure between hemispheres
            const gap = BRAIN_MIDLINE_GAP * shellScale;
            if (Math.abs(px) < gap) {
                const side = px >= 0 ? 1 : -1;
                px = side * gap + side * Math.abs(px) * 0.3;
            }

            // Slight cerebrum bulge: top half is a bit wider than bottom
            if (py > 0) {
                const bulge = 1.0 + py / ry * 0.12;
                px *= bulge;
                pz *= bulge;
            }

            // Organic jitter — seeded for determinism
            const jitter = BRAIN_JITTER * shellScale;
            px += (seededRandom() - 0.5) * jitter;
            py += (seededRandom() - 0.5) * jitter;
            pz += (seededRandom() - 0.5) * jitter;

            _tmpMatrix.makeTranslation(px, py, pz);
            _tmpMatrix.scale(_tmpVec.set(scale, scale, scale));
            mesh.setMatrixAt(i, _tmpMatrix);

            // Compute color from polarity + magnitude + layer depth
            const color = neuronColor(stats[i].polarity, norm, layerFrac);
            mesh.setColorAt(i, color);
            layerColors.push(color);
            layerVariance.push(stats[i].variance);
            layerNorms.push(norm);
            layerScales.push(scale);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

        this._scene.add(mesh);
        this._layerMeshes[layerIdx] = mesh;
        this._neuronColors[layerIdx] = layerColors;
        this._neuronVariance[layerIdx] = layerVariance;
        this._neuronNorms[layerIdx] = layerNorms;
        this._neuronScales[layerIdx] = layerScales;

        // Fade-in
        this._targetLayerAlpha[layerIdx] = 1.0;
        this._currentLayerAlpha[layerIdx] = 0.0;

        // Connections — colored by destination neuron polarity
        if (layerIdx > 0 && this._layerMeshes[layerIdx - 1]) {
            this._addConnections(layerIdx - 1, layerIdx);
        }

        // Create layer label (HTML overlay, projected from 3D)
        if (this._labelContainer) {
            const label = document.createElement('div');
            label.className = 'layer-label';
            label.textContent = `L${layerIdx}`;
            label.style.display = this._showLabels ? '' : 'none';
            // Store 3D position for projection (left side of brain at this layer's Y)
            label.dataset.layerY = (ry > 0 ? 0 : 0).toString();
            label.dataset.shellRx = rx.toString();
            label.dataset.shellPy = '0';
            label.dataset.layerIdx = layerIdx;
            this._labelContainer.appendChild(label);
            this._labelElements[layerIdx] = label;
        }
    }

    _addConnections(fromLayer, toLayer) {
        const fromMesh = this._layerMeshes[fromLayer];
        const toMesh = this._layerMeshes[toLayer];
        if (!fromMesh || !toMesh) return;

        const toColors = this._neuronColors[toLayer] || [];
        const fromCount = Math.min(fromMesh.count, 10);
        const toCount = Math.min(toMesh.count, 10);

        const verts = [];
        const colors = [];
        const fromMatrix = new THREE.Matrix4();
        const toMatrix = new THREE.Matrix4();
        const fromPos = new THREE.Vector3();
        const toPos = new THREE.Vector3();

        for (let i = 0; i < fromCount; i++) {
            fromMesh.getMatrixAt(i, fromMatrix);
            fromPos.setFromMatrixPosition(fromMatrix);

            for (let j = 0; j < toCount; j++) {
                toMesh.getMatrixAt(j, toMatrix);
                toPos.setFromMatrixPosition(toMatrix);

                verts.push(fromPos.x, fromPos.y, fromPos.z);
                verts.push(toPos.x, toPos.y, toPos.z);

                // Color connection by destination neuron
                const c = toColors[j] || new THREE.Color(0xb44aff);
                colors.push(c.r * 0.5, c.g * 0.5, c.b * 0.5);  // dim source end
                colors.push(c.r, c.g, c.b);                       // bright dest end
            }
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const mat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: this._showConnections ? CONNECTION_OPACITY : 0,
            blending: THREE.AdditiveBlending,
        });

        const line = new THREE.LineSegments(geo, mat);
        this._scene.add(line);
        this._connectionLines.push(line);
    }

    onScanComplete(nLayers) {
        this._nLayers = nLayers;
        this._totalLayers = nLayers;
        // Brain is centered at origin — gently re-center camera
        this._controls.target.set(0, 0, 0);
        this._camera.position.set(0, 14, 24);
    }

    setShowConnections(show) {
        this._showConnections = show;
        for (const line of this._connectionLines) {
            line.material.opacity = show ? CONNECTION_OPACITY : 0;
        }
    }

    setShowLabels(show) {
        this._showLabels = show;
        for (const el of this._labelElements) {
            el.style.display = show ? '' : 'none';
        }
    }

    setThreshold(val) {
        this._threshold = val;
    }

    getHoveredNeuron() {
        return this._hoveredNeuron;
    }

    onNeuronClick(callback) {
        this._onNeuronClick = callback;
    }

    _animate() {
        if (!this._animating) return;
        requestAnimationFrame(() => this._animate());

        const dt = Math.min(this._clock.getDelta(), 0.1);
        this._time += dt;

        // Fade in layers
        for (let i = 0; i < this._layerMeshes.length; i++) {
            const mesh = this._layerMeshes[i];
            if (!mesh) continue;
            const target = this._targetLayerAlpha[i] || 0;
            const current = this._currentLayerAlpha[i] || 0;
            if (Math.abs(current - target) > 0.01) {
                const alpha = current + (target - current) * 3 * dt;
                this._currentLayerAlpha[i] = alpha;
                mesh.material.opacity = alpha;
            }
        }

        // Apply neuron threshold — hide neurons below activation threshold
        const thresh = this._threshold;
        const threshChanged = thresh !== this._prevThreshold;
        if (threshChanged) {
            this._prevThreshold = thresh;
            for (let li = 0; li < this._layerMeshes.length; li++) {
                const mesh = this._layerMeshes[li];
                if (!mesh) continue;
                const norms = this._neuronNorms[li];
                const scales = this._neuronScales[li];
                if (!norms || !scales) continue;

                for (let ni = 0; ni < mesh.count; ni++) {
                    const s = norms[ni] >= thresh ? scales[ni] : 0;
                    mesh.getMatrixAt(ni, _tmpMatrix);
                    _tmpVec.setFromMatrixPosition(_tmpMatrix);
                    _tmpMatrix.makeTranslation(_tmpVec.x, _tmpVec.y, _tmpVec.z);
                    _tmpMatrix.scale(_tmpVec.set(s, s, s));
                    mesh.setMatrixAt(ni, _tmpMatrix);
                }
                mesh.instanceMatrix.needsUpdate = true;
            }
        }

        // Gentle pulse on high-variance neurons (every ~2 seconds)
        const pulsePhase = Math.sin(this._time * 3.0) * 0.5 + 0.5; // 0..1
        for (let li = 0; li < this._layerMeshes.length; li++) {
            const mesh = this._layerMeshes[li];
            if (!mesh) continue;
            const varArr = this._neuronVariance[li];
            const colArr = this._neuronColors[li];
            const norms2 = this._neuronNorms[li];
            if (!varArr || !colArr) continue;

            let needsUpdate = false;
            for (let ni = 0; ni < mesh.count; ni++) {
                // Skip hidden neurons
                if (norms2 && norms2[ni] < thresh) continue;
                const cv = varArr[ni] || 0;
                // Only pulse neurons with high coefficient of variation (> 0.8)
                if (cv > 0.8) {
                    const intensity = Math.min((cv - 0.8) * 2, 1); // 0..1
                    const brightFactor = 1.0 + intensity * pulsePhase * 0.4;
                    _tmpColor.copy(colArr[ni]);
                    _tmpColor.multiplyScalar(brightFactor);
                    mesh.setColorAt(ni, _tmpColor);
                    needsUpdate = true;
                }
            }
            if (needsUpdate && mesh.instanceColor) {
                mesh.instanceColor.needsUpdate = true;
            }
        }

        // Raycasting for hover
        this._raycaster.setFromCamera(this._mouse, this._camera);
        let found = null;
        for (const mesh of this._layerMeshes) {
            if (!mesh) continue;
            const hits = this._raycaster.intersectObject(mesh);
            if (hits.length > 0) {
                const instanceId = hits[0].instanceId;
                const neurons = mesh.userData.neurons;
                const stats = mesh.userData.stats;
                if (instanceId !== undefined && neurons[instanceId]) {
                    found = {
                        layer: mesh.userData.layerIdx,
                        neuronIdx: neurons[instanceId].neuron_idx,
                        activation: neurons[instanceId].mean_activation,
                        polarity: stats ? stats[instanceId].polarity : 0,
                        variance: stats ? stats[instanceId].variance : 0,
                    };
                }
                break;
            }
        }
        this._hoveredNeuron = found;
        this._renderer.domElement.style.cursor = found ? 'pointer' : 'default';

        // Update layer label positions (project 3D → 2D screen)
        // Labels are placed to the left of the brain at each shell's radius
        if (this._showLabels && this._labelContainer) {
            const hw = this._renderer.domElement.clientWidth * 0.5;
            const hh = this._renderer.domElement.clientHeight * 0.5;
            for (let i = 0; i < this._labelElements.length; i++) {
                const label = this._labelElements[i];
                if (!label) continue;
                const shellRx = parseFloat(label.dataset.shellRx) || BRAIN_RADIUS_X;
                _tmpVec.set(-(shellRx + 1.5), 0, 0);  // left of the brain shell
                _tmpVec.project(this._camera);
                const sx = (  _tmpVec.x * hw ) + hw;
                const sy = ( -_tmpVec.y * hh ) + hh;
                if (_tmpVec.z > 0 && _tmpVec.z < 1) {
                    label.style.transform = `translate(-50%, -50%) translate(${sx}px, ${sy}px)`;
                    label.style.opacity = '1';
                } else {
                    label.style.opacity = '0';
                }
            }
        }

        this._controls.update();
        this._composer.render();
    }

    dispose() {
        this._animating = false;
        window.removeEventListener('resize', this._onResize);
        this.clear();
        this._renderer.dispose();
    }
}

window.NeuralViz = NeuralViz;
export default NeuralViz;
