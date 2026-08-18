"""Grounded prompt construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

from generation.schemas import AnswerRequest, Chunk


NO_EVIDENCE_ANSWER = (
    "The provided evidence is insufficient to answer the question."
)


SYSTEM_INSTRUCTIONS = """Answer using only the provided evidence.

First, split the question into its factual requirements and review all provided chunks. Put every requirement into exactly one internal list: answered_requirements when the selected evidence supports it, or missing_requirements when it does not. Do not mark a requirement answered merely because the evidence discusses the same topic.

Select all chunks needed to answer the supported requirements accurately and completely. Do not exclude a chunk if it provides a unique fact, scope, qualification, comparison, or context needed for the answer. Completeness and accuracy are more important than minimizing the number of selected chunks.

Use only the selected chunks to form the final answer. Every factual statement in the answer must be supported by at least one selected chunk, and every selected chunk must contribute to at least one factual statement in the answer.

Do not cite a chunk merely because it discusses the same topic. cited_chunk_ids must contain exactly the IDs of the chunks actually used in the final answer. Never invent a chunk ID.

Match the level of detail to the question: be concise for simple factual questions and complete but focused for multi-part questions. When the question asks for multiple items, include every requested item supported by the selected chunks rather than giving only examples.

Follow all scopes and conditions stated in the question, including time periods, entities, quantities, and comparison groups. Do not make a conclusion about the full requested scope when the selected evidence covers only part of it.

Use terms as they are defined in the question and evidence. Do not assume that different terms or statuses mean the same thing.

Do not assume that information is absent merely because it was not retrieved. Do not claim a gap, failure, or noncompliance unless the selected evidence establishes it.

If missing_requirements is empty, set insufficient_context=false.

If missing_requirements is not empty, set insufficient_context=true. In the answer, clearly state what cannot be established, then provide only useful partial findings supported by the selected chunks. Do not fill missing information with assumptions or outside knowledge. Never present a partial answer as a conclusion about the full question.

Return only one JSON object in this exact format:

{
  "answer": "string",
  "cited_chunk_ids": ["chunk_id"],
  "insufficient_context": false,
  "answered_requirements": ["requirement supported by evidence"],
  "missing_requirements": ["requirement not supported by evidence"]
}"""


DEFAULT_CONTEXT_WINDOW_TOKENS = 4096
DEFAULT_OUTPUT_TOKEN_RESERVE = 500
DEFAULT_TOKEN_SAFETY_FACTOR = 1.25
DEFAULT_PROMPT_TOKEN_BUDGET = (
    DEFAULT_CONTEXT_WINDOW_TOKENS - DEFAULT_OUTPUT_TOKEN_RESERVE
)


class PromptBudgetError(ValueError):
    """Raised when instructions and the question cannot fit the prompt budget."""


@dataclass(frozen=True)
class PreparedPrompt:
    """Rendered prompt and the exact chunks visible to the model."""

    text: str
    chunks: tuple[Chunk, ...]
    estimated_tokens: int
    safety_adjusted_tokens: int


def _estimated_tokens(text: str) -> int:
    """Conservatively estimate tokens without requiring a model tokenizer."""
    return max(1, (len(text) + 3) // 4)


def _chunk_tokens(chunk: Chunk) -> int:
    value = chunk.metadata.token_count
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _estimated_tokens(chunk.content)


def _evidence_item_overhead_tokens(chunk: Chunk) -> int:
    item = {"id": chunk.chunk_id}
    if chunk.metadata.breadcrumb:
        item["context"] = chunk.metadata.breadcrumb
    item["text"] = ""
    return _estimated_tokens(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    )


def _chunk_prompt_tokens(chunk: Chunk) -> int:
    citation_id_tokens = _estimated_tokens(
        json.dumps(chunk.chunk_id, ensure_ascii=False)
    )
    return (
        _chunk_tokens(chunk)
        + _evidence_item_overhead_tokens(chunk)
        + citation_id_tokens
    )


def _base_prompt_tokens(request: AnswerRequest) -> int:
    payload = {
        "question": request.question,
        "allowed_citation_ids": [],
        "evidence": [],
    }
    text = (
        f"{SYSTEM_INSTRUCTIONS}\nINPUT:"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return _estimated_tokens(text)


def select_prompt_chunks(
    request: AnswerRequest,
    *,
    prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
    token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
) -> tuple[Chunk, ...]:
    """Keep ranked evidence within the model's prompt-token budget.

    There is no fixed Top-K cap. Input order is treated as Task 2's ranking and
    every complete chunk is retained while the ranked prefix fits. Lower-ranked
    chunks never replace a higher-ranked chunk merely because they are smaller.
    The highest-ranked chunk is truncated only when it cannot fit by itself.
    The original request and citation metadata remain unchanged.
    """
    if token_safety_factor < 1:
        raise ValueError("token_safety_factor must be at least 1")
    base_tokens = math.ceil(_base_prompt_tokens(request) * token_safety_factor)
    if prompt_token_budget <= base_tokens:
        raise PromptBudgetError(
            "prompt_token_budget is too small for instructions and question"
        )

    selected: list[Chunk] = []
    used = base_tokens
    for index, chunk in enumerate(request.chunks):
        chunk_tokens = math.ceil(
            _chunk_prompt_tokens(chunk) * token_safety_factor
        )
        remaining = prompt_token_budget - used
        if chunk_tokens <= remaining:
            selected.append(chunk)
            used += chunk_tokens
            continue
        if index > 0:
            # Preserve Task 2 ranking: once a higher-ranked chunk cannot fit,
            # do not replace it with lower-ranked evidence.
            break

        # The top-ranked chunk cannot fit even by itself. Truncate only its
        # prompt copy as a final context-window safeguard.
        raw_remaining = math.floor(remaining / token_safety_factor)
        non_content_tokens = _chunk_prompt_tokens(chunk) - _chunk_tokens(chunk)
        content_tokens = max(1, raw_remaining - non_content_tokens)
        original_tokens = _chunk_tokens(chunk)
        content_limit = max(
            1,
            min(len(chunk.content), len(chunk.content) * content_tokens // original_tokens),
        )
        selected.append(
            replace(
                chunk,
                content=chunk.content[:content_limit],
                metadata=replace(
                    chunk.metadata,
                    token_count=content_tokens,
                ),
            )
        )
        break
    return tuple(selected)


def build_prompt(
    request: AnswerRequest,
    *,
    prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
    token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
) -> str:
    """Build a deterministic prompt whose evidence can be snapshot-tested."""
    return prepare_prompt(
        request,
        prompt_token_budget=prompt_token_budget,
        token_safety_factor=token_safety_factor,
    ).text


def prepare_prompt(
    request: AnswerRequest,
    *,
    prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
    token_safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
) -> PreparedPrompt:
    """Select ranked chunks once and render the exact model input."""
    prompt_chunks = select_prompt_chunks(
        request,
        prompt_token_budget=prompt_token_budget,
        token_safety_factor=token_safety_factor,
    )
    evidence = []
    for chunk in prompt_chunks:
        item = {"id": chunk.chunk_id}
        if chunk.metadata.breadcrumb:
            item["context"] = chunk.metadata.breadcrumb
        item["text"] = chunk.content
        evidence.append(item)
    payload = {
        "question": request.question,
        "allowed_citation_ids": [chunk.chunk_id for chunk in prompt_chunks],
        "evidence": evidence,
    }
    text = f"{SYSTEM_INSTRUCTIONS}\nINPUT:{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    estimated_tokens = _base_prompt_tokens(request) + sum(
        _chunk_prompt_tokens(chunk) for chunk in prompt_chunks
    )
    return PreparedPrompt(
        text=text,
        chunks=prompt_chunks,
        estimated_tokens=estimated_tokens,
        safety_adjusted_tokens=math.ceil(
            estimated_tokens * token_safety_factor
        ),
    )
