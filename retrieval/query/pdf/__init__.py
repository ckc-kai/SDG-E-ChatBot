"""Public PDF/Excel evidence retrieval interfaces."""

from retrieval.query.pdf.query import (
    EVIDENCE_GROUPS,
    EvidenceGroup,
    EvidenceRetrievalResult,
    RankedResult,
    retrieve,
    retrieve_configured,
    retrieve_evidence,
)

__all__ = [
    "EVIDENCE_GROUPS",
    "EvidenceGroup",
    "EvidenceRetrievalResult",
    "RankedResult",
    "retrieve",
    "retrieve_configured",
    "retrieve_evidence",
]
