"""
QueryExpander — optionally augments a QueryBuilder result list with
LLM-generated title/synonym variants.

Design rules:
  - ONLY used during query-generation phase.
  - The LLM receives the query string, NOT any job page content.
  - When LLM is unavailable the expander falls back silently to the
    rule-based variants already produced by QueryBuilder.
  - No embeddings, no vector operations.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client protocol (dependency-inverted so tests can inject a mock)
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient(Protocol):
    """Minimal interface expected from an LLM client."""

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        """Send prompt, return raw response text."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Expansion prompt template
# ─────────────────────────────────────────────────────────────────────────────

_EXPANSION_PROMPT = """\
You are a job search assistant.
Given the base job search query below, generate up to 4 alternative phrasing
variants that a recruiter might use on LinkedIn, Indeed, or Naukri.

Rules:
- Keep the same location (if present).
- Vary only the job title using common synonyms or seniority terms.
- Do NOT add extra skills or qualifications not present in the original.
- Return a JSON array of strings, no commentary, no markdown.

Base query: {base_query}

Example output:
["Python Developer Mumbai", "Python Engineer Mumbai", "Backend Developer Python Mumbai"]
"""


# ─────────────────────────────────────────────────────────────────────────────
# QueryExpander
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpander:
    """
    Optionally enriches a list of query strings with LLM-generated variants.

    Usage:
        expander = QueryExpander(llm_client=my_client)  # real usage
        expander = QueryExpander()                        # fallback / tests
        expanded = expander.expand(base_queries, max_total=5)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        max_llm_variants: int = 4,
    ) -> None:
        self._llm = llm_client
        self._max_llm_variants = max_llm_variants

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def expand(
        self,
        base_queries: List[str],
        *,
        max_total: int = 5,
    ) -> List[str]:
        """
        Merge base_queries with LLM variants, deduplicating.

        Args:
            base_queries: Queries already produced by QueryBuilder.
            max_total:    Cap on total returned queries.

        Returns:
            Deduplicated list, base queries first, LLM variants appended.
        """
        if not base_queries:
            return []

        merged = list(base_queries)
        seen = {q.lower() for q in merged}

        if self._llm is not None and len(merged) < max_total:
            llm_variants = self._call_llm(base_queries[0])
            for variant in llm_variants:
                key = variant.lower().strip()
                if key and key not in seen:
                    merged.append(variant.strip())
                    seen.add(key)
                if len(merged) >= max_total:
                    break

        return merged[:max_total]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _call_llm(self, base_query: str) -> List[str]:
        """
        Call the LLM and parse the JSON array response.
        Returns an empty list on any failure.
        """
        if self._llm is None:
            return []

        prompt = _EXPANSION_PROMPT.format(base_query=base_query)
        try:
            raw = self._llm.complete(prompt, max_tokens=256)
            # Strip markdown code fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            variants = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError("LLM did not return a JSON array")
            return [str(v) for v in variants[: self._max_llm_variants]]
        except Exception as exc:
            logger.warning("QueryExpander LLM call failed, skipping: %s", exc)
            return []
