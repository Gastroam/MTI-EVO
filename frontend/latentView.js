/**
 * MTI-EVO Latent Space Immersion
 * Renders high-dimensional embeddings as a 3D Point Cloud.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createLatentView({ canvas, log }) {
    if (!canvas) return { setEmbeddings: () => { }, start: () => { } };

    // 1. Scene Setup
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#05080a');
    scene.fog = new THREE.FogExp2('#05080a', 0.02);

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 40);

    // 2. Controls
    let controls = null;
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // 3. Point Cloud
    const cloudGroup = new THREE.Group();
    scene.add(cloudGroup);

    let pointsMesh = null;
    let hoverMesh = null;

    // Interaction
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.5;
    const mouse = new THREE.Vector2();

    function setAttractors(data) {
        let attractors = [];
        if (Array.isArray(data)) {
            attractors = data;
        } else if (data && data.attractors && Array.isArray(data.attractors)) {
            attractors = data.attractors;
        } else {
            log("[LatentView] ⚠️ Invalid attractor data format received.");
            return;
        }

        if (attractors.length === 0) {
            log("[LatentView] No attractors to render.");
            return;
        }
        cloudGroup.clear();

        const count = attractors.length;
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const sizes = new Float32Array(count);

        const familyColors = {
            "Pillar": new THREE.Color(0xff0000),   // Red
            "Resonant": new THREE.Color(0x00ffff), // Cyan
            "Bridge": new THREE.Color(0xffff00),   // Yellow
            "Ghost": new THREE.Color(0x333333),     // Dark Gray
            "Culture": new THREE.Color(0xFFD700)    // Gold
        };

        const familyZ = {
            "Pillar": 20,
            "Resonant": 10,
            "Culture": 5,   // Interleaved with logic
            "Bridge": 0,
            "Ghost": -10
        };

        attractors.forEach((a, i) => {
            // Map Mass/Reach to X/Y
            // Mass (0-100) -> X (-20 to 20)
            const x = (a.mass - 50) * 0.8;

            // Reach (0-100) -> Y (-20 to 20)
            const y = (a.reach_min - 50) * 0.8;

            // Family -> Z Tier
            const z = familyZ[a.family] || 0;

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            const c = familyColors[a.family] || new THREE.Color(0xffffff);
            colors[i * 3] = c.r;
            colors[i * 3 + 1] = c.g;
            colors[i * 3 + 2] = c.b;

            // Size based on Mass
            sizes[i] = Math.max(0.5, a.mass / 20);
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        // Attach data for raycasting
        geometry.userData = { nodes: attractors };

        // Material
        const texture = createGlowTexture();
        const material = new THREE.PointsMaterial({
            size: 15.0, // Increased size
            vertexColors: true,
            map: texture,
            transparent: true,
            opacity: 1.0,
            blending: THREE.AdditiveBlending, // Glow effect
            depthWrite: false,
            sizeAttenuation: true
        });

        pointsMesh = new THREE.Points(geometry, material);
        cloudGroup.add(pointsMesh);

        // Hover Indicator
        if (hoverMesh) {
            scene.remove(hoverMesh); // Cleanup old mesh
            hoverMesh.geometry.dispose();
            hoverMesh.material.dispose();
        }

        const hoverGeo = new THREE.BufferGeometry();
        hoverGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
        const hoverTexture = createGlowTexture();
        hoverMesh = new THREE.Points(hoverGeo, new THREE.PointsMaterial({
            color: 0xffffff,
            size: 25.0,
            map: hoverTexture,
            transparent: true,
            opacity: 1.0,
            blending: THREE.AdditiveBlending,
            depthTest: false
        }));
        hoverMesh.visible = false;
        scene.add(hoverMesh);

        // Add connecting lines (Topology)
        addConnections(attractors, positions);

        log(`[LatentView] Rendered ${count} attractors.`);
    }

    function addConnections(attractors, positions) {
        // Simple logic: connect items in same family or specfic pairs
        // For now, just a visual wireframe box or center line
        const material = new THREE.LineBasicMaterial({ color: 0x334455, transparent: true, opacity: 0.2 });
        // ... (Optional: add topology lines if needed)
    }

    function createGlowTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64; canvas.height = 64;
        const ctx = canvas.getContext('2d');

        // Soft Glow Gradient
        const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
        grad.addColorStop(0.2, 'rgba(255, 255, 255, 0.8)');
        grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.2)');
        grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);

        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }

    // Loop
    function animate() {
        requestAnimationFrame(animate);
        if (controls) controls.update();

        if (cloudGroup) {
            cloudGroup.rotation.y += 0.001; // Slow drift
        }

        // Hover logic
        raycaster.setFromCamera(mouse, camera);
        if (pointsMesh) {
            const intersects = raycaster.intersectObject(pointsMesh);
            if (intersects.length > 0) {
                const index = intersects[0].index;
                const nodes = pointsMesh.geometry.userData.nodes;
                const node = nodes[index];

                // Position highlighter
                const posA = pointsMesh.geometry.attributes.position;
                hoverMesh.position.set(
                    posA.getX(index),
                    posA.getY(index),
                    posA.getZ(index)
                );
                // Transform to world for correct visual
                hoverMesh.position.applyMatrix4(pointsMesh.matrixWorld);
                hoverMesh.visible = true;

                // Update HUD (reuse existing target HUD function if possible, or emit event)
                // For now, simple log or custom event
                window.dispatchEvent(new CustomEvent('latent-hover', { detail: node }));
            } else {
                hoverMesh.visible = false;
                window.dispatchEvent(new CustomEvent('latent-hover', { detail: null }));
            }
        }

        renderer.render(scene, camera);
    }

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    });

    function start() {
        animate();
    }

    return { setAttractors, start };
}
