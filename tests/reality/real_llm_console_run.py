"""
Real LLM Console Test Runner
============================
Runs real local inference via MTI-EVO's LLMAdapter (GGUF engine),
prints logs to console, and writes reproducible artifacts to disk.

Usage:
  python tests/reality/real_llm_console_run.py
  python tests/reality/real_llm_console_run.py --model-path H:\\models\\gemma-3-4b-it-q4_0.gguf
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime


def _configure_console_encoding() -> None:
    # Prevent Windows cp1252 crashes from unicode-rich logs.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mti_evo.llm_adapter import LLMAdapter  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_mti_config() -> dict:
    cfg_path = ROOT / "mti_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_paths(explicit: str | None) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))

    cfg = _load_mti_config()
    cfg_model = cfg.get("model_path")
    if cfg_model:
        p = pathlib.Path(cfg_model)
        candidates.append(p)
        # Common local correction for paths like H:\models\MTI-EVO\models\... -> H:\models\...
        corrected = str(cfg_model).replace("\\MTI-EVO\\models\\", "\\")
        if corrected != cfg_model:
            candidates.append(pathlib.Path(corrected))

    candidates.extend(
        [
            pathlib.Path(r"H:\models\gemma-3-4b-it-q4_0.gguf"),
            pathlib.Path(r"H:\models\meta-llama-3-8b-instruct.Q4_K_M.gguf"),
            pathlib.Path(r"H:\models\gemma-3-12b-it-Q2_K_L.gguf"),
        ]
    )
    return candidates


def _resolve_model_path(explicit: str | None) -> pathlib.Path:
    for path in _candidate_paths(explicit):
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "No GGUF model found. Pass --model-path explicitly (example: H:\\models\\gemma-3-4b-it-q4_0.gguf)."
    )


def _prompts() -> list[str]:
    return [
        "Explain in one paragraph why deterministic seeds matter in neural experiments.",
        "Describe how eviction pressure impacts long-term memory stability in sparse attractor systems.",
        "Give a concise method section paragraph for reporting reproducible LLM experiments.",
    ]


def run(model_path: pathlib.Path, max_tokens: int, gpu_layers: int, n_ctx: int, n_batch: int) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "tests" / "reality" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"real_llm_run_{ts}.json"
    txt_path = out_dir / f"real_llm_run_{ts}.txt"

    run_info = {
        "timestamp": ts,
        "model_path": str(model_path),
        "engine": "gguf",
        "n_ctx": n_ctx,
        "gpu_layers": gpu_layers,
        "max_tokens": max_tokens,
        "max_tokens": max_tokens,
        "n_batch": n_batch,
        "samples": [],
    }

    log("Starting REAL LLM test run.")
    log(f"Model path: {model_path}")
    log(f"Artifacts: {json_path} and {txt_path}")

    config = {
        "model_type": "gguf",
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "gpu_layers": gpu_layers,
        "temperature": 0.7,
    }

    t0 = time.perf_counter()
    adapter = LLMAdapter(config=config, auto_load=True)
    load_ms = (time.perf_counter() - t0) * 1000
    log(f"Adapter loaded. backend={adapter.backend} load_ms={load_ms:.2f}")

    transcript_lines = [
        f"REAL LLM RUN {ts}",
        f"Model: {model_path}",
        f"Backend: {adapter.backend}",
        f"Load ms: {load_ms:.2f}",
        "",
    ]

    failures = 0
    for idx, prompt in enumerate(_prompts(), start=1):
        log(f"Prompt {idx}: {prompt}")
        response = adapter.infer(prompt, max_tokens=max_tokens, stop=["<end_of_turn>"])
        text = (response.text or "").strip()
        ok = bool(text) and "Not Loaded" not in text and not text.startswith("Error:")
        if not ok:
            failures += 1

        log(
            f"Result {idx}: ok={ok} latency_ms={response.latency_ms:.2f} "
            f"tokens={response.tokens} chars={len(text)}"
        )
        log(f"Response {idx} preview: {text[:240]}")

        run_info["samples"].append(
            {
                "index": idx,
                "prompt": prompt,
                "ok": ok,
                "latency_ms": response.latency_ms,
                "tokens": response.tokens,
                "coherence": response.coherence,
                "gpu_stats": response.gpu_stats,
                "response_text": text,
            }
        )

        transcript_lines.extend(
            [
                f"--- SAMPLE {idx} ---",
                f"PROMPT: {prompt}",
                f"OK: {ok}",
                f"LATENCY_MS: {response.latency_ms:.2f}",
                f"TOKENS: {response.tokens}",
                "RESPONSE:",
                text,
                "",
            ]
        )

    adapter.unload_model()
    log("Adapter unloaded.")

    latencies = [s["latency_ms"] for s in run_info["samples"]]
    run_info["summary"] = {
        "sample_count": len(run_info["samples"]),
        "failure_count": failures,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "all_passed": failures == 0,
    }

    json_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text("\n".join(transcript_lines), encoding="utf-8")

    log(
        "Run complete. "
        f"all_passed={run_info['summary']['all_passed']} "
        f"avg_latency_ms={run_info['summary']['avg_latency_ms']:.2f}"
    )
    log(f"Wrote: {json_path}")
    log(f"Wrote: {txt_path}")

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real MTI-EVO LLM test with visible console logs.")
    parser.add_argument("--model-path", default=None, help="Path to a GGUF model.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens per prompt.")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers for GGUF engine.")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context window.")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size for prompt processing.")
    args = parser.parse_args()

    try:
        model_path = _resolve_model_path(args.model_path)
    except Exception as exc:
        log(f"FAILED before run: {exc}")
        return 2

    return run(
        model_path=model_path,
        max_tokens=args.max_tokens,
        gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
