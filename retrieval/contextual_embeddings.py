"""Contextual-document formatting shared by ingestion and in-place refresh."""

from __future__ import annotations

# v3: the leading Document line now carries the manifest's readable title rather
# than a filename, so every previously written contextual vector is stale.
CONTEXTUAL_EMBEDDING_RECIPE = "document-section-chunk-v3-title"


def contextual_embedding_text(
    document_title: str,
    breadcrumb: str | None,
    content: str,
) -> str:
    """Compose embedding-only context without changing stored display content.

    ``document_title`` is the manifest's ``display_title`` where one exists, and
    the filename otherwise.
    """
    parts = [f"Document: {document_title}"]
    if breadcrumb:
        parts.append(f"Section: {breadcrumb}")
    parts.append(f"Chunk: {content}")
    return "\n".join(parts)


def contextual_embedding_text_for_model(
    document_title: str,
    breadcrumb: str | None,
    content: str,
    tokenizer,
    max_seq_length: int,
) -> str:
    """Compose context while guaranteeing that stored chunk text is not truncated.

    The complete document/section/chunk form is preferred. If it exceeds the
    model limit, the supplementary document-title line is removed first. The full
    breadcrumb and chunk content are retained; the final guard raises instead of
    silently truncating. Against a 32k-token embedder this fallback should never
    fire, but it costs nothing and it is the check that would catch a chunk-size
    or model change that quietly outgrew the window.
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

    contextual_text = contextual_embedding_text(document_title, breadcrumb, content)
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
            f"document title line ({token_count} > {max_seq_length})."
        )
    return contextual_text
