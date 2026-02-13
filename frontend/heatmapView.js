import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createHeatmapView({ canvas, log }) {
    if (!canvas) {
        if (log) log("HeatmapView: missing canvas");
        return;
    }

    log("HeatmapView: Initializing Three.js scene...");

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    // scene.background = new THREE.Color(0x050505); // Dark background
    // Transparent background to blend with UI
    scene.background = null;

    // Add subtle fog for depth
    scene.fog = new THREE.FogExp2(0x050505, 0.02);

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.set(20, 20, 30);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    function animate() {
        requestAnimationFrame(animate);

        // Auto-resize check every frame to handle tab switches
        resize();

        if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;

        controls.update();
        renderer.render(scene, camera);
    }



    // 2. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xff00ff, 1.5); // Cyberpunk magenta
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0x00ffff, 1.0); // Cyan rim
    dirLight2.position.set(-10, 10, -20);
    scene.add(dirLight2);

    // 3. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2; // Don't go below ground

    // 4. Grid / Floor
    const gridHelper = new THREE.GridHelper(50, 50, 0x334155, 0x1e293b);
    scene.add(gridHelper);

    // 5. Data Representation (InstancedMesh for perf)
    let mesh = null;
    const dummy = new THREE.Object3D();

    // [VISUALS] REVERT to StandardMaterial for "Nice" look + Emissive
    const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.3,
        metalness: 0.8,
        vertexColors: true,
        emissive: 0x000000,
        emissiveIntensity: 0.5
    });

    // Handle Resize
    function resize() {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (w === 0 || h === 0 || w === canvas.width && h === canvas.height) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
    }
    window.addEventListener('resize', resize);

    // Public API
    function renderData({ neurons, timeBins, matrix }) {
        if (mesh) {
            scene.remove(mesh);
            mesh.geometry.dispose();
        }

        const count = neurons.length * timeBins.length;
        const geometry = new THREE.BoxGeometry(0.8, 1, 0.8);
        geometry.translate(0, 0.5, 0);

        mesh = new THREE.InstancedMesh(geometry, material, count);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        // Store metadata for updates
        mesh.userData = { neurons, timeBins };

        updateMesh(matrix);
        scene.add(mesh);
    }

    function updateData(matrix) {
        if (!mesh) return;
        updateMesh(matrix);
    }

    function updateMesh(matrix) {
        const { neurons, timeBins } = mesh.userData;
        const color = new THREE.Color();
        let idx = 0;
        const xOffset = -(timeBins.length * 1.0) / 2;
        const zOffset = -(neurons.length * 1.0) / 2;

        for (let i = 0; i < neurons.length; i++) {
            for (let j = 0; j < timeBins.length; j++) {
                const val = matrix[i][j];

                dummy.position.set(xOffset + j, 0, zOffset + i);
                const h = Math.max(0.1, val * 12);
                dummy.scale.set(1, h, 1);
                dummy.updateMatrix();
                mesh.setMatrixAt(idx, dummy.matrix);

                // [VISUALS] High Visibility Palette
                // Use explicit Hex for Low values to ensure they aren't black
                if (val < 0.2) color.setHex(0x3b27ba); // Bright Purple-Blue (Lighter than before)
                else if (val < 0.5) color.setHSL(0.6, 1.0, 0.5); // Pure Blue
                else if (val < 0.8) color.setHSL(0.8, 1.0, 0.6); // Magenta/Pink
                else color.setHSL(0.1, 1.0, 0.85); // White-Yellow

                mesh.setColorAt(idx, color);
                idx++;
            }
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    return { renderData, updateData, start: animate };
}
