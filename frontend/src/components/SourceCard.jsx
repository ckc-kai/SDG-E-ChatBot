function SourceCard({ source }) {
    const location = source.sheet
      ? `${source.sheet}${source.row_start ? ` · rows ${source.row_start}–${source.row_end}` : ''}`
      : source.page_start !== null && source.page_start !== undefined
        ? `p.${source.page_start}–${source.page_end}`
        : 'Location unavailable';

    return (
      <div className="source-tab">
        <span className="source-tab-badge">Source</span>
        <p className="source-tab-title">{source.source_pdf}</p>
        <span className="source-tab-ref">
          {location}{source.breadcrumb ? ` · ${source.breadcrumb}` : ''}
        </span>
      </div>
    );
  }
  
  export default SourceCard;
