# 04. INTERFACE PROTOCOL (Frontend V2)

## 1. Overview
The MTI-EVO Frontend (Interface) serves as the visual cortex of the system, translating high-dimensional neural states into a human-perceptible Cyberpunk HUD. This document outlines the V2 specifications, including the "Tech Panel" layout system and data integrity protocols.

## 2. Design Philosophy
*   **Aesthetic**: "Functional Cyberpunk" - High contrast, dark glass, neon accents (Cyan/Magenta), and chamfered geometry.
*   **Typography**: Standardized on **Rajdhani** (Tech/Clean) for all text, with **Orbitron** used sparingly for headers/values.
*   **Color**: 
    *   Text: Pure White (`#FFFFFF`) for maximum legibility.
    *   Backgrounds: Deep Dark Glass (`rgba(5, 8, 12, 0.9)`).
    *   Accents: Cyan (`#00f3ff`) for active elements, Magenta (`#ff0055`) for alerts.

## 3. Component Architecture

### 3.1 Tech Panel System (`.tech-panel-framed`)
All information widgets are encapsulated in modular containers akin to hardware modules.
*   **Geometry**: 45-degree chamfered corners using CSS `clip-path`.
*   **Frame**: Decorative border accents (brackets) generated via pseudo-elements.
*   **Header**: Standardized `.tech-header` with label and specialized ID code (e.g., `SYS.01`).

### 3.2 HUD Widgets
*   **Cognitive Drift**: Visualizes the balance between Logic and Dream states.
*   **Core Reactor**: Circular gauge monitoring VRAM usage and Tensor Flux.
*   **Neural Oscillation**: Real-time waveform visualization of system activity.
*   **Network Status**: Simple FPS/Ping vitals.

## 4. Data Integrity Protocol
To ensure the interface reflects reality (no "movie magic"):
1.  **Strict Polling**: The frontend polls `/api/status` every 1000ms.
2.  **No Mock Data**: If the API is unreachable, the HUD explicitly displays `OFFLINE` / `NO SIGNAL`.
3.  **Auto-Recovery**: Use of robust `try/catch` and removal of guard clauses ensures the UI automatically reconnects when the core comes online.

## 5. Technical Stack
*   **Core**: Vanilla JS (ES6+), HTML5 Canvas.
*   **3D**: Three.js for Neural Topology visualization.
*   **Styling**: Pure CSS3 with Variables (no preprocessors needed).
