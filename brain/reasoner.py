"""
BrainReasoner — local GGUF reasoning via llama.cpp.

This build runs in text-reasoning mode by default so it does not depend on
LLaVA-specific multimodal handlers that can mismatch model families.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from brain.prompts import CHAT_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger("brain.reasoner")


def _bootstrap_llama_runtime() -> None:
    """Register CUDA DLL locations before importing llama.cpp on Windows."""
    if not hasattr(os, "add_dll_directory"):
        return

    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception as exc:
        log.debug("Torch CUDA DLL bootstrap skipped: %s", exc)


class BrainReasoner:
    """Reasoning via llama.cpp GGUF in robust text mode."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_gpu_layers: int = 99,
        n_ctx: int = 4096,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._model = None
        self._degraded = False
        self._inference_lock = threading.Lock()

        log.info("Loading Brain model: %s", model_path)
        log.info("  n_gpu_layers=%d, n_ctx=%d", n_gpu_layers, n_ctx)
        _bootstrap_llama_runtime()

        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
            log.info("Brain model loaded successfully (llama.cpp text mode)")

        except Exception as e:
            log.error("Brain model load failed: %s", e)
            self._degraded = True
            log.warning("Brain running in fallback rule mode (no LLM)")

    @property
    def available(self) -> bool:
        return self._model is not None or self._degraded

    def analyze(
        self,
        frame: np.ndarray,
        detection_summary: str,
        audio_transcript: str = "",
        interaction_context: str = "",
        camera_id: str = "main",
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a frame with the VLM Brain.

        Args:
            frame: BGR camera frame
            detection_summary: YOLO detection text
            audio_transcript: What the person said (from STT)
            camera_id: camera identifier

        Returns:
            Parsed JSON verdict or None on failure.
        """
        if not self.available:
            log.error("Brain not available")
            return None

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        user_text = build_user_prompt(
            timestamp=timestamp,
            camera_id=camera_id,
            detection_summary=detection_summary,
            audio_transcript=audio_transcript,
            interaction_context=interaction_context,
        )

        if self._model is None:
            return self._fallback_verdict(detection_summary, audio_transcript)

        verdict = self._infer(user_text)
        if verdict is None:
            log.warning("Brain output was unusable; falling back to deterministic rules")
            return self._fallback_verdict(detection_summary, audio_transcript)
        return verdict

    def chat(
        self,
        user_message: str,
        frame: Optional[np.ndarray] = None,
        context: str = "",
    ) -> str:
        """
        Interactive chat mode — user sends text, Brain responds.
        Used by the browser command console.
        """
        if not self.available:
            return "Brain not available."
        if self._model is None:
            return (
                "Brain is running in fallback mode (LLM unavailable). "
                "I can still react to detections and dispatch basic actions."
            )

        messages: List[Dict] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        ]

        prompt = user_message
        if context:
            prompt = f"[Context: {context}]\n\n{user_message}"
        if frame is not None:
            prompt = (
                "[Note: image input unavailable in current runtime; "
                "responding from text context only]\n\n" + prompt
            )
        messages.append({"role": "user", "content": prompt})

        try:
            with self._inference_lock:
                response = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
            raw = response["choices"][0]["message"]["content"]
            return self._chat_text_from_raw(raw)
        except Exception as e:
            log.error("Chat failed: %s", e)
            return f"Error: {e}"

    # ── Internal inference ───────────────────────────────────────────

    def _infer(self, user_text: str) -> Optional[Dict]:
        """Run text inference via llama.cpp."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        t0 = time.perf_counter()
        try:
            with self._inference_lock:
                response = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    top_p=0.9,
                )
            elapsed = time.perf_counter() - t0

            raw = response["choices"][0]["message"]["content"]
            tokens = response.get("usage", {}).get("completion_tokens", 0)
            log.info("Brain inference: %.2fs (%d tokens)", elapsed, tokens)

            return self._parse_output(raw)

        except Exception as e:
            log.error("Brain inference failed: %s", e)
            return None

    # ── JSON Parsing ─────────────────────────────────────────────────

    def _parse_output(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse the Brain's raw output into structured JSON."""
        if not raw:
            return None

        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Fix LaTeX-escaped underscores (common LLM artifact)
        cleaned = cleaned.replace("\\_", "_")

        # Fix trailing commas (common LLM mistake)
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

        # Attempt 1: Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract JSON object from surrounding text
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                candidate = json_match.group()
                candidate = candidate.replace("\\_", "_")
                candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Attempt 3: Salvage truncated JSON
        try:
            truncated = cleaned.rstrip()
            if not truncated.endswith("}"):
                open_braces = truncated.count("{") - truncated.count("}")
                open_brackets = truncated.count("[") - truncated.count("]")
                if truncated.count('"') % 2 != 0:
                    truncated += '"'
                truncated += "]" * max(0, open_brackets)
                truncated += "}" * max(0, open_braces)
            truncated = re.sub(r',\s*([\]}])', r'\1', truncated)
            return json.loads(truncated)
        except Exception:
            pass

        log.warning("Failed to parse Brain output: %s", raw[:300])
        return None

    def _chat_text_from_raw(self, raw: str) -> str:
        """Convert structured Brain output into a conversational chat reply."""
        verdict = self._parse_output(raw)
        if not verdict:
            return raw

        actions = verdict.get("actions", [])
        speak_action = next((a for a in actions if a.get("function") == "speak"), None)
        if speak_action:
            message = speak_action.get("params", {}).get("message", "").strip()
            if message:
                return message

        alert_action = next((a for a in actions if a.get("function") == "alert"), None)
        if alert_action:
            message = alert_action.get("params", {}).get("message", "").strip()
            if message:
                return message

        reasoning = str(verdict.get("chain_of_thought", "")).strip()
        if reasoning:
            return reasoning

        return raw

    def _fallback_verdict(self, detection_summary: str, audio_transcript: str) -> Dict[str, Any]:
        """Simple deterministic fallback when LLM model cannot be loaded."""
        text = f"{detection_summary} {audio_transcript}".lower()
        high_risk_terms = ["knife", "gun", "weapon", "crowbar", "bat", "intruder", "break", "forced"]
        medium_risk_terms = ["person", "car", "motorcycle", "truck", "bus"]

        threat = 0.25
        if any(t in text for t in high_risk_terms):
            threat = 0.85
        elif any(t in text for t in medium_risk_terms):
            threat = 0.55

        actions: List[Dict[str, Any]] = []
        if threat >= 0.8:
            actions.append({
                "function": "speak",
                "params": {"message": "Warning. Suspicious activity detected. Authorities may be notified."},
            })
            actions.append({
                "function": "alert",
                "params": {"message": f"High threat detected: {detection_summary}", "priority": "critical"},
            })
        elif threat >= 0.5:
            actions.append({
                "function": "speak",
                "params": {"message": "Attention. Activity detected in monitored area."},
            })
            actions.append({
                "function": "log",
                "params": {"event": f"Medium-risk activity: {detection_summary}"},
            })
        else:
            actions.append({
                "function": "log",
                "params": {"event": f"Low-risk activity: {detection_summary}"},
            })

        return {
            "threat_score": threat,
            "chain_of_thought": "Fallback rule mode active (LLM unavailable).",
            "actions": actions,
        }
