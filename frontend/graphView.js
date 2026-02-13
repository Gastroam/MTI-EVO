/**
 * MTI Brain 3D Monitor - Graph Visualization
 * Renders neurons as glowing spheres and connections as animated lines.
 */
import * as THREE from 'three';

export function createGraphView({ scene }) {
    const group = new THREE.Group();
    scene.add(group);

    const nodes = new Map(); // id -> {mesh, data}
    const edges = [];
    const labelSprites = [];

    // Materials
    const createNodeMaterial = () => new THREE.MeshStandardMaterial({
        color: 0x3b82f6,
        emissive: 0x1e40af,
        emissiveIntensity: 0.3,
        metalness: 0.3,
        roughness: 0.6
    });

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0x64748b,
        transparent: true,
        opacity: 0.6
    });

    /**
     * Create a single neuron node
     */
    function makeNode(id, pos, degree = 1) {
        const radius = Math.min(0.5 + degree * 0.03, 1.5);
        const geometry = new THREE.SphereGeometry(radius, 24, 24);
        const material = createNodeMaterial();
        const mesh = new THREE.Mesh(geometry, material);

        mesh.position.set(pos[0], pos[1], pos[2]);
        mesh.userData = {
            id,
            degree,
            activation: 0.0,
            baseScale: radius,
            baseY: pos[1]
        };

        group.add(mesh);
        nodes.set(id, { mesh, data: mesh.userData });
    }

    /**
     * Create an edge between two nodes
     */
    function makeEdge(sourceId, targetId, weight = 0.5) {
        const a = nodes.get(sourceId)?.mesh;
        const b = nodes.get(targetId)?.mesh;
        if (!a || !b) return;

        const points = [a.position.clone(), b.position.clone()];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = edgeMaterial.clone();
        material.opacity = 0.3 + weight * 0.5;

        const line = new THREE.Line(geometry, material);
        line.userData = { source: sourceId, target: targetId, weight };
        group.add(line);
        edges.push(line);
    }

    /**
     * Load a complete graph topology
     */
    function loadGraph(nodeList, edgeList) {
        // Clear existing
        for (const { mesh } of nodes.values()) {
            group.remove(mesh);
        }
        nodes.clear();

        edges.forEach((e) => group.remove(e));
        edges.length = 0;

        // Create nodes
        nodeList.forEach(n => makeNode(n.id, n.pos, n.degree || 1));

        // Create edges
        edgeList.forEach(e => makeEdge(e.source, e.target, e.weight || 0.5));
    }

    /**
     * Set activation level for a node (0-1)
     */
    function setActivation(id, value) {
        const n = nodes.get(id);
        if (!n) return;

        n.data.activation = value;
        const m = n.mesh.material;

        // Color: blue (cold) → orange (hot)
        const hue = 0.6 - (0.5 * value); // 0.6 = blue, 0.1 = orange
        m.color.setHSL(hue, 0.8, 0.5);
        m.emissive.setHSL(hue, 0.9, 0.3);
        m.emissiveIntensity = 0.2 + 0.6 * value;

        // Scale pulse
        const scale = n.data.baseScale * (1.0 + 0.3 * value);
        n.mesh.scale.setScalar(scale);
    }

    /**
     * Per-frame animation tick
     */
    function tick(dt) {
        const t = performance.now() * 0.001;

        // Subtle floating animation for nodes
        for (const { mesh, data } of nodes.values()) {
            const float = Math.sin(t * 0.5 + data.baseY) * 0.1;
            mesh.position.y = data.baseY + float;

            // Gentle rotation
            mesh.rotation.y += dt * 0.1;
        }

        // Edge opacity pulsing
        edges.forEach((line) => {
            const w = line.userData.weight ?? 0.5;
            const pulse = 0.5 + 0.5 * Math.sin(t * w * 2);
            line.material.opacity = 0.3 + 0.4 * pulse * w;
        });
    }

    // Subscribe to render loop
    scene.subscribe(tick);

    return { loadGraph, setActivation, nodes, edges };
}
