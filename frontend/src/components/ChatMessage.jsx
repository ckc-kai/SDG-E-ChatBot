import SourceCard from './SourceCard';

function ChatMessage({ role, content, sources, onSave, saved }) {
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
        {sources && sources.length > 0 && (
          <div className="sources-row">
            {sources.map((source) => (
              <SourceCard key={`${source.doc_id}-${source.page_start}`} source={source} />
            ))}
          </div>
        )}
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