import SourceCard from './SourceCard';

function ChatMessage({
  role,
  content,
  citations,
  requestId,
  onSave,
  saved,
}) {
  if (role === 'user') {
    return (
      <div className="message-row user">
        <div className="bubble user">{content}</div>
      </div>
    );
  }

  return (
    <div className="message-row assistant">
      <div className="bubble assistant">
        {content}
        {citations && citations.length > 0 && (
          <div className="sources-row">
            {citations.map((citation) => (
              <SourceCard key={citation.chunk_id} source={citation} />
            ))}
          </div>
        )}
        {requestId && <span className="request-id">Request: {requestId}</span>}
      </div>
      {onSave && (
        <button className="save-button" onClick={onSave} disabled={saved}>
          {saved ? '★ Saved' : '☆ Save this answer'}
        </button>
      )}
    </div>
  );
}

export default ChatMessage;
