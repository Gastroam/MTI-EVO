/**
 * MTI-EVO Advanced Monitor - HIVE Network View
 * Three.js visualization of HIVE mesh topology.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createHiveView({ canvas, log }) {
    if (!canvas) return { loadGraph: () => { }, updateFlows: () => { }, start: () => { } };

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0f14');

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.set(0, 10, 20);
    camera.lookAt(0, 0, 0);

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, 0.8);
    const directional = new THREE.DirectionalLight(0xffffff, 1.0);
    directional.position.set(10, 20, 10);
    scene.add(ambient, directional);

    // Controls
    let controls = null;
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Groups
    const nodesGroup = new THREE.Group();
    const edgesGroup = new THREE.Group();
    const flowsGroup = new THREE.Group();
    scene.add(nodesGroup, edgesGroup, flowsGroup);

    const nodeMap = new Map();

    function loadGraph(nodes, edges) {
        // Clear existing
        nodesGroup.clear();
        edgesGroup.clear();
        nodeMap.clear();

        if (!nodes) return;

        // Create nodes
        nodes.forEach((n, i) => {
            const geometry = new THREE.SphereGeometry(0.5, 16, 16);
            const material = new THREE.MeshStandardMaterial({
                color: n.role === 'master' ? 0xf59e0b : 0x3b82f6,
                emissive: 0x1e40af,
                emissiveIntensity: 0.3
            });
            const mesh = new THREE.Mesh(geometry, material);

            // Position in circle
            const angle = (i / nodes.length) * Math.PI * 2;
            const radius = 8;
            mesh.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
            mesh.userData = n;

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
                    color: e.type === 'control' ? 0xf59e0b : 0x64748b,
                    transparent: true,
                    opacity: 0.6
                });
                const line = new THREE.Line(geometry, material);
                edgesGroup.add(line);
            });
        }

        log?.(`HIVE graph loaded: ${nodes.length} nodes`);
    }

    function updateFlows(flows) {
        // Clear existing flows
        flowsGroup.clear();

        if (!flows) return;

        // Create animated flow particles
        flows.forEach(f => {
            const a = nodeMap.get(f.from);
            const b = nodeMap.get(f.to);
            if (!a || !b) return;

            const geometry = new THREE.SphereGeometry(0.15, 8, 8);
            const material = new THREE.MeshBasicMaterial({
                color: f.status === 'ok' ? 0x22c55e : 0xef4444
            });
            const particle = new THREE.Mesh(geometry, material);
            particle.position.copy(a.position);
            particle.userData = { from: a.position.clone(), to: b.position.clone(), progress: 0 };

            flowsGroup.add(particle);
        });
    }

    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);

        if (controls) controls.update();

        // Animate flow particles
        const dt = clock.getDelta();
        flowsGroup.children.forEach(particle => {
            particle.userData.progress += dt * 0.5;
            if (particle.userData.progress >= 1) {
                particle.userData.progress = 0;
            }
            const t = particle.userData.progress;
            particle.position.lerpVectors(particle.userData.from, particle.userData.to, t);
        });

        // Rotate nodes slightly
        nodesGroup.children.forEach(m => {
            m.rotation.y += dt * 0.2;
        });

        renderer.render(scene, camera);
    }

    function start() {
        animate();
    }

    // Resize
    window.addEventListener('resize', () => {
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    });

    return { loadGraph, updateFlows, start };
}
