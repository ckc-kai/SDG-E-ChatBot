function SourceCard({ source }) {
    return (
      <div className="source-tab">
        <span className="source-tab-badge">{source.content_type}</span>
        <p className="source-tab-title">{source.source_pdf}</p>
        <span className="source-tab-ref">
          p.{source.page_start}–{source.page_end} · {source.breadcrumb}
        </span>
        <p className="source-tab-snippet">{source.snippet}</p>
      </div>
    );
  }
  
  export default SourceCard;