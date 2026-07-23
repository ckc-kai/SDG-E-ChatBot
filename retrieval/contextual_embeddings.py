"""Contextual-document formatting shared by ingestion and in-place refresh."""

from __future__ import annotations

CONTEXTUAL_EMBEDDING_RECIPE = "document-section-chunk-v2-budgeted"


def contextual_embedding_text(
    source_pdf: str,
    breadcrumb: str | None,
    content: str,
) -> str:
    """Compose embedding-only context without changing stored display content."""
    parts = [f"Document: {source_pdf}"]
    if breadcrumb:
        parts.append(f"Section: {breadcrumb}")
    parts.append(f"Chunk: {content}")
    return "\n".join(parts)


def contextual_embedding_text_for_model(
    source_pdf: str,
    breadcrumb: str | None,
    content: str,
    tokenizer,
    max_seq_length: int,
) -> str:
    """Compose context while guaranteeing that stored chunk text is not truncated.

    The complete document/section/chunk form is preferred. If it exceeds the
    model limit, the supplementary source-filename line is removed first. The
    full breadcrumb and chunk content are retained. Current 400-token chunks fit
    after that fallback; the final guard raises instead of silently truncating.
    """
    def bounded_token_count(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length + 1,
            )
        )

    contextual_text = contextual_embedding_text(source_pdf, breadcrumb, content)
    if bounded_token_count(contextual_text) <= max_seq_length:
        return contextual_text

    parts = []
    if breadcrumb:
        parts.append(f"Section: {breadcrumb}")
    parts.append(f"Chunk: {content}")
    contextual_text = "\n".join(parts)
    token_count = bounded_token_count(contextual_text)
    if token_count > max_seq_length:
        raise ValueError(
            "Contextual embedding input exceeds the model limit even without the "
            f"document line ({token_count} > {max_seq_length})."
        )
    return contextual_text
