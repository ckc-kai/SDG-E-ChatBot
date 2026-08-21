import { useState } from 'react';
import SourceCard from './SourceCard';

function Sidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  saved,
  onRemoveSaved,
}) {
  const [tab, setTab] = useState('chats');
  const [expandedSavedIndex, setExpandedSavedIndex] = useState(null);

  const toggleSaved = (i) => {
    setExpandedSavedIndex((prev) => (prev === i ? null : i));
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <button
          className={`sidebar-tab${tab === 'chats' ? ' sidebar-tab-active' : ''}`}
          onClick={() => setTab('chats')}
        >
          Chats
        </button>
        <button
          className={`sidebar-tab${tab === 'saved' ? ' sidebar-tab-active' : ''}`}
          onClick={() => setTab('saved')}
        >
          Saved
        </button>
      </div>

      {tab === 'chats' && (
        <div className="sidebar-panel">
          <button className="new-chat-button" onClick={onNewChat}>
            + New chat
          </button>
          {conversations.length === 0 && (
            <p className="sidebar-empty">No conversations yet.</p>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`chat-list-item${
                conv.id === activeConversationId ? ' chat-list-item-active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <span className="chat-list-item-title">
                {conv.title || 'New chat'}
              </span>
              <button
                className="chat-list-item-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(conv.id);
                }}
                aria-label="Delete conversation"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {tab === 'saved' && (
        <div className="sidebar-panel">
          {saved.length === 0 && (
            <p className="sidebar-empty">Nothing saved yet — star an answer to keep it here.</p>
          )}
          {saved.map((item, i) => (
            <div className="saved-item" key={i}>
              <p
                className="saved-item-question"
                onClick={() => toggleSaved(i)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && toggleSaved(i)}
              >
                {item.question}
              </p>
              {expandedSavedIndex === i && (
                <div className="saved-item-detail">
                  <p className="saved-item-answer">{item.answer}</p>
                  {item.sources && item.sources.length > 0 && (
                    <div className="sources-row">
                      {item.sources.map((source) => (
                        <SourceCard key={`${source.doc_id}-${source.page_start}`} source={source} />
                      ))}
                    </div>
                  )}
                </div>
              )}
              <button className="saved-item-remove" onClick={() => onRemoveSaved(i)}>
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

export default Sidebar;