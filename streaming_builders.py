from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class _ToolCallBuilder:
    id: Optional[str] = None
    name: Optional[str] = None
    type: str = "function"
    args_chunks: List[str] = field(default_factory=list)

    def append_args(self, piece: Optional[str]) -> None:
        if piece is None:
            return
        self.args_chunks.append(piece)

    @property
    def args_text(self) -> str:
        return "".join(self.args_chunks)


class ToolCallStream:
    def __init__(self, fallback_name: str = "auto_tool") -> None:
        self.builders: Dict[int, _ToolCallBuilder] = {}
        self._fallback_name = fallback_name
        self._last_index: Optional[int] = None

    def on_tool_calls_delta(self, tool_calls: list[dict]) -> None:
        for tc in (tool_calls or []):
            idx = int(tc.get("index", 0))
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            name = fn.get("name")
            piece = fn.get("arguments")

            # Decide target builder index with robust coalescing of null-id fragments
            target_idx = idx
            if not tc_id:
                if idx in self.builders:
                    # If builder at idx has no content and no name, prefer last active builder
                    b_cur = self.builders[idx]
                    if (not b_cur.args_chunks) and (not b_cur.name) and (self._last_index is not None):
                        target_idx = self._last_index
                elif self._last_index is not None:
                    # No builder at idx; continue current builder
                    target_idx = self._last_index

            b = self.builders.get(target_idx) or _ToolCallBuilder()
            if tc_id:
                b.id = tc_id
            if name:
                b.name = name
            b.append_args(piece)
            self.builders[target_idx] = b
            self._last_index = target_idx

    def on_function_call_delta(self, function_call: dict) -> None:
        idx = self._last_index if self._last_index is not None else 0
        b = self.builders.get(idx) or _ToolCallBuilder()
        name = function_call.get("name")
        if name:
            b.name = name
        piece = function_call.get("arguments") or ""
        b.append_args(piece)
        self.builders[idx] = b
        self._last_index = idx

    def finalize(self) -> list[dict] | None:
        if not self.builders:
            return None
        ts = int(time.time() * 1000)
        out: list[dict] = []
        for idx in sorted(self.builders.keys()):
            b = self.builders[idx]
            args_emit = b.args_text or ""
            # Do not drop content even if not valid JSON; callers can repair
            out.append({
                "index": idx,
                "id": b.id or f"call_{idx}_{ts}",
                "type": "function",
                "function": {
                    "name": b.name or self._fallback_name,
                    "arguments": args_emit,
                },
            })
        self.builders.clear()
        return out
