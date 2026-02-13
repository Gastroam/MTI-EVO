/**
 * MTI Brain 3D Monitor - Scene Manager
 * Initializes Three.js, camera, lights, and render loop.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createScene({ canvas }) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0f14');
    scene.fog = new THREE.FogExp2('#0b0f14', 0.015);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 15, 35);
    camera.lookAt(0, 0, 0);

    // Lighting
    const ambient = new THREE.AmbientLight(0x404060, 0.8);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(15, 25, 20);
    const fill = new THREE.DirectionalLight(0x8080ff, 0.4);
    fill.position.set(-10, 10, -15);
    const rim = new THREE.PointLight(0x3b82f6, 0.8);
    rim.position.set(0, -10, 0);
    scene.add(ambient, key, fill, rim);

    // Grid helper for reference
    const grid = new THREE.GridHelper(50, 50, 0x1f2937, 0x111827);
    grid.position.y = -5;
    scene.add(grid);

    // Orbit Controls
    let controls = null;
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 10;
    controls.maxDistance = 100;

    // Resize handler
    const onResize = () => {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
    };
    window.addEventListener('resize', onResize);
    onResize();

    // Subscriber pattern for per-frame updates
    const subscribers = new Set();
    const subscribe = (fn) => subscribers.add(fn);
    const unsubscribe = (fn) => subscribers.delete(fn);

    const clock = new THREE.Clock();

    const start = () => {
        const loop = () => {
            const dt = clock.getDelta();

            if (controls) controls.update();

            subscribers.forEach((fn) => fn(dt));
            renderer.render(scene, camera);
            requestAnimationFrame(loop);
        };
        loop();
    };

    return { scene, camera, renderer, start, subscribe, unsubscribe, add: (obj) => scene.add(obj) };
}
