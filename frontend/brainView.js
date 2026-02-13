/**
 * MTI-EVO Advanced Monitor - Brain View
 * Three.js 3D visualization of neurons with activation.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export function createBrainView({ canvas, log }) {
    if (!canvas) return { loadGraph: () => { }, setActivation: () => { }, start: () => { } };

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0f14');
    scene.fog = new THREE.FogExp2('#0b0f14', 0.012);

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.set(0, 15, 35);
    camera.lookAt(0, 0, 0);

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, 0.8);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(15, 25, 20);
    const fill = new THREE.DirectionalLight(0x8080ff, 0.4);
    fill.position.set(-10, 10, -15);
    scene.add(ambient, key, fill);

    // Grid
    const grid = new THREE.GridHelper(50, 50, 0x1f2937, 0x111827);
    grid.position.y = -5;
    scene.add(grid);

    // Controls
    let controls = null;
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 10;
    controls.maxDistance = 100;

    // Groups
    const nodesGroup = new THREE.Group();
    const edgesGroup = new THREE.Group();
    const dustGroup = new THREE.Group(); // Neuro-Dust
    scene.add(dustGroup, edgesGroup, nodesGroup);

    // [AR] Neuro-Dust Field
    const dustGeo = new THREE.BufferGeometry();
    const dustCount = 2000;
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount * 3; i++) {
        dustPos[i] = (Math.random() - 0.5) * 100;
    }
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    const dustMat = new THREE.PointsMaterial({
        color: 0x00f3ff,
        size: 0.1,
        transparent: true,
        opacity: 0.4
    });
    const dustSystem = new THREE.Points(dustGeo, dustMat);
    dustGroup.add(dustSystem);

    const nodeMap = new Map();

    const domainColors = {
        math: 0xf59e0b,
        bio: 0x22c55e,
        vision: 0xa855f7,
        astro: 0x06b6d4,
        unknown: 0x3b82f6
    };

    function loadGraph(nodes, edges) {
        // Clear existing
        nodesGroup.clear();
        edgesGroup.clear();
        nodeMap.clear();

        if (!nodes || nodes.length === 0) {
            log?.('No nodes to load');
            return;
        }

        // Create nodes
        nodes.forEach(n => {
            const radius = Math.min(0.3 + (n.degree || 1) * 0.02, 1.2);
            const geometry = new THREE.SphereGeometry(radius, 24, 24);

            const color = domainColors[n.domain] || domainColors.unknown;
            const material = new THREE.MeshStandardMaterial({
                color,
                emissive: color,
                emissiveIntensity: 0.2,
                metalness: 0.3,
                roughness: 0.6
            });

            const mesh = new THREE.Mesh(geometry, material);

            // Position from data or random
            if (n.pos && Array.isArray(n.pos)) {
                mesh.position.set(n.pos[0], n.pos[1], n.pos[2]);
            } else {
                mesh.position.set(
                    (Math.random() - 0.5) * 20,
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 20
                );
            }

            mesh.userData = {
                id: n.id,
                label: n.label,
                seed: n.seed,
                domain: n.domain,
                baseScale: radius,
                baseY: mesh.position.y,
                activation: 0
            };

            nodesGroup.add(mesh);
            nodeMap.set(n.id, mesh);
        });

        // Create edges
        if (edges) {
            edges.forEach(e => {
                const a = nodeMap.get(e.source);
                const b = nodeMap.get(e.target);
                if (!a || !b) return;

                const points = [a.position.clone(), b.position.clone()];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: 0x64748b,
                    transparent: true,
                    opacity: 0.3 + (e.weight || 0.5) * 0.4
                });
                const line = new THREE.Line(geometry, material);
                line.userData = { weight: e.weight };
                edgesGroup.add(line);
            });
        }

        log?.(`Brain graph loaded: ${nodes.length} nodes, ${edges?.length || 0} edges`);
    }

    function setActivation(nodeId, value) {
        const mesh = nodeMap.get(nodeId);
        if (!mesh) return;

        mesh.userData.activation = value;

        // Color shift: blue (cold) → orange (hot)
        const hue = 0.6 - (0.5 * value);
        mesh.material.color.setHSL(hue, 0.8, 0.5);
        mesh.material.emissive.setHSL(hue, 0.9, 0.3);
        mesh.material.emissiveIntensity = 0.2 + 0.6 * value;

        // Scale
        const scale = mesh.userData.baseScale * (1.0 + 0.4 * value);
        mesh.scale.setScalar(scale);

        // [PHASE 27] Show label on high activation
        if (value > 0.7) {
            showLabel(mesh, mesh.userData.label || mesh.userData.id);
        } else {
            hideLabel(mesh);
        }
    }

    // Label Logic
    const labelContainer = document.getElementById('brain-labels') || createLabelContainer();

    function createLabelContainer() {
        const div = document.createElement('div');
        div.id = 'brain-labels';
        div.style.position = 'absolute';
        div.style.top = '0';
        div.style.left = '0';
        div.style.width = '100%';
        div.style.height = '100%';
        div.style.pointerEvents = 'none';
        div.style.overflow = 'hidden';
        canvas.parentElement.appendChild(div);
        return div;
    }

    function showLabel(mesh, text) {
        if (!mesh.userData.labelEl) {
            const el = document.createElement('div');
            el.className = 'brain-label';
            el.textContent = text;
            el.style.position = 'absolute';
            el.style.color = '#4ade80';
            el.style.fontSize = '12px';
            el.style.fontFamily = 'monospace';
            el.style.padding = '2px 6px';
            el.style.background = 'rgba(0,0,0,0.7)';
            el.style.borderRadius = '4px';
            labelContainer.appendChild(el);
            mesh.userData.labelEl = el;
        }

        // Update position in loop
        mesh.userData.labelVisible = true;
    }

    function hideLabel(mesh) {
        if (mesh.userData.labelEl) {
            mesh.userData.labelEl.remove();
            mesh.userData.labelEl = null;
        }
        mesh.userData.labelVisible = false;
    }

    // [BLOOM] Post-Processing Setup
    const renderScene = new RenderPass(scene, camera);

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
        1.5,  // strength
        0.4,  // radius
        0.85  // threshold
    );
    bloomPass.threshold = 0;
    bloomPass.strength = 2.0; // High glow for neon look
    bloomPass.radius = 0.5;

    let composer = null;
    composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    // Anti-aliasing pass (FXAA) or SMAA could be added if script loaded, 
    // but for now we rely on hardware AA/high pixel ratio or just the glow masking aliasing.

    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);

        const dt = clock.getDelta();
        const t = performance.now() * 0.001;

        if (controls) controls.update();

        // [AR] Quantized Dust Rotation (Drift)
        if (dustGroup) {
            dustGroup.rotation.y = t * 0.02;
            dustGroup.rotation.x = Math.sin(t * 0.1) * 0.05;
        }

        // Float animation
        nodesGroup.children.forEach(mesh => {
            const float = Math.sin(t * 0.5 + mesh.userData.baseY) * 0.1;
            mesh.position.y = mesh.userData.baseY + float;
            mesh.rotation.y += dt * 0.1;

            // Update Label Position
            if (mesh.userData.labelVisible && mesh.userData.labelEl) {
                const pos = mesh.position.clone();
                pos.project(camera);

                const x = (pos.x * .5 + .5) * canvas.clientWidth;
                const y = (-(pos.y * .5) + .5) * canvas.clientHeight;

                // Hide if behind camera
                if (pos.z < 1) {
                    mesh.userData.labelEl.style.display = 'block';
                    mesh.userData.labelEl.style.transform = `translate(-50%, -100%) translate(${x}px, ${y - 10}px)`;
                } else {
                    mesh.userData.labelEl.style.display = 'none';
                }
            }
        });

        // Edge pulse
        edgesGroup.children.forEach(line => {
            const w = line.userData.weight || 0.5;
            const pulse = 0.5 + 0.5 * Math.sin(t * w * 2);
            line.material.opacity = 0.3 + 0.4 * pulse * w;
        });

        // Render via Composer (Bloom) or fallback
        if (composer) {
            composer.render();
        } else {
            renderer.render(scene, camera);
        }
    }

    // [AR] Raycaster for Targeting
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredNode = null;

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(nodesGroup.children);

        if (intersects.length > 0) {
            const target = intersects[0].object;
            if (hoveredNode !== target) {
                hoveredNode = target;
                updateTargetHUD(target.userData);
            }
        } else {
            if (hoveredNode) {
                hoveredNode = null;
                updateTargetHUD(null);
            }
        }
    });

    function trackNodeLoop() {
        // if (hoveredNode) updateTooltip(); // Removed: redundant with HUD overlay
        requestAnimationFrame(trackNodeLoop);
    }
    trackNodeLoop();

    function updateTargetHUD(data) {
        const hud = document.getElementById('hudTarget');
        if (!hud) return;

        if (data) {
            hud.style.display = 'block';
            const idStr = String(data.id || '?');
            const seedStr = String(data.seed || '?');

            hud.querySelector('.target-header').textContent = `TGT: ${data.label || idStr.substring(0, 8)}`;
            hud.querySelector('.target-body').innerHTML = `
                <span style="color:#8899a6">ID:</span>   ${idStr.substring(0, 8)}...<br>
                <span style="color:#8899a6">SEED:</span> <span style="color:#00ff41">${seedStr.substring(0, 12)}</span><br>
                <span style="color:#8899a6">DOM:</span>  ${data.domain || 'UNK'}<br>
                <span style="color:#8899a6">ACT:</span>  <span style="color:${(data.activation || 0) > 0.5 ? '#ff0055' : '#00f3ff'}">${(data.activation || 0).toFixed(3)}</span>
            `;
        } else {
            hud.style.display = 'none';
        }
    }

    // Resize
    const onResize = () => {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (w === 0 || h === 0) return;

        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
        composer?.setSize(w, h);
    };
    window.addEventListener('resize', onResize);



    function start() {
        animate();
    }

    return { loadGraph, setActivation, start };
}
