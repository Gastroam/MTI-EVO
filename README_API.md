# MTI-EVO API Reference
**Version**: 1.0 | **Port**: 8800 (default)

This document describes the HTTP API exposed by MTI-EVO for frontend integration.

---

## Quick Start
```bash
# Start the server
python run_evo.py --port 8800
```

---

## Endpoints

### `GET /status`
Returns the current health and statistics of the MTI Brain.

**Response:**
```json
{
  "status": "online",
  "neurons": 74,
  "version": "2.0 (Stabilized)",
  "mode": "Bicameral"
}
```

---

### `POST /control/dream`
Triggers a Hebbian Associative Drift (Dream) starting from a seed concept.

**Request Body:**
```json
{
  "seed": "consciousness",
  "steps": 10
}
```

**Response:**
```json
{
  "seed": "consciousness",
  "drift_length": 10,
  "path": ["consciousness", "self", "memory", "time", "god", "..."]
}
```

---

### `POST /control/interview`
Performs a "Cognitive Interview" — asks the AI to explain its own associations.

**Request Body:**
```json
{
  "target": "love"
}
```

**Response:**
```json
{
  "target": "love",
  "associations": ["37543329", "hope", "light", "vitality"],
  "explanation": "My subconscious links 'love' with building blocks of emotional connection.",
  "latency_ms": 450
}
```

---

### `POST /v1/local/reflex` (Legacy)
Direct Telepathy Bridge for raw LLM inference. Maintained for backward compatibility.

**Request Body:**
```json
{
  "action": "telepathy",
  "prompt": "Explain the concept of recursion.",
  "max_tokens": 150
}
```

**Response:**
```json
{
  "response": "Recursion is a programming technique where a function calls itself...",
  "stats": {
    "latency_ms": 320,
    "tokens": 45
  }
}
```

---

## CORS
All endpoints include `Access-Control-Allow-Origin: *` for frontend integration.

---

## Architecture
```
┌─────────────────────────────────────────────────┐
│           Frontend (Web/Desktop)                │
└─────────────────────┬───────────────────────────┘
                      │ HTTP (Port 8800)
                      ▼
┌─────────────────────────────────────────────────┐
│           MTI-EVO Control Plane (server.py)     │
├─────────────────────────────────────────────────┤
│  /status         │  /control/dream              │
│  /control/interview │ /v1/local/reflex          │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│  EvoAPI       │           │  LLMAdapter   │
│  (api.py)     │           │  (Gemma/Qwen) │
│  - Hebbian    │           │  - Telepathy  │
│  - Dreams     │           │  - Inference  │
└───────────────┘           └───────────────┘
```

---

## Future Extensions
- **WebSocket**: Real-time dream streaming.
- **Batch Interview**: `/control/interview/batch` for 20-question protocols.
- **Metrics Export**: Prometheus-compatible `/metrics` endpoint.
