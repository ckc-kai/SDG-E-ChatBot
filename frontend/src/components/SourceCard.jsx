function SourceCard({ source }) {
  const hasPdfPage = source.page_start != null && source.page_end != null;
  const hasExcelLocation = source.sheet || source.row_start != null;

  return (
    <div className="source-tab">
      <p className="source-tab-title">{source.source_pdf || 'Source'}</p>
      {hasPdfPage && (
        <span className="source-tab-ref">
          p.{source.page_start + 1}–{source.page_end} · {source.breadcrumb}
        </span>
      )}
      {hasExcelLocation && (
        <span className="source-tab-ref">
          {source.sheet}
          {source.row_start != null ? `, row ${source.row_start}` : ''} · {source.breadcrumb}
        </span>
      )}
      {!hasPdfPage && !hasExcelLocation && source.breadcrumb && (
        <span className="source-tab-ref">{source.breadcrumb}</span>
      )}
    </div>
  );
}

export default SourceCard;