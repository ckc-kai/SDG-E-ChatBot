import { useEffect, useState } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import Sidebar from './components/Sidebar';
import { askQuestion, checkHealth } from './api/client';

const STORAGE_KEY_CONVERSATIONS = 'sdge_chat_conversations';
const STORAGE_KEY_SAVED = 'sdge_chat_saved';

function makeId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function makeEmptyConversation() {
  return { id: makeId(), title: '', messages: [] };
}

function loadFromStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function App() {
  const [conversations, setConversations] = useState(() => {
    const stored = loadFromStorage(STORAGE_KEY_CONVERSATIONS, null);
    return stored && stored.length > 0 ? stored : [makeEmptyConversation()];
  });
  const [activeConversationId, setActiveConversationId] = useState(
    () => conversations[0].id
  );
  const [saved, setSaved] = useState(() => loadFromStorage(STORAGE_KEY_SAVED, []));
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);

  useEffect(() => {
    checkHealth().then(setBackendOnline).catch(() => setBackendOnline(false));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_CONVERSATIONS, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_SAVED, JSON.stringify(saved));
  }, [saved]);

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) || conversations[0];

  const updateActiveConversation = (updater) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === activeConversationId ? updater(c) : c))
    );
  };

  const handleNewChat = () => {
    if (activeConversation.messages.length === 0) return;
    const fresh = makeEmptyConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveConversationId(fresh.id);
  };

  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
  };

  const handleDeleteConversation = (id) => {
    setConversations((prev) => {
      const remaining = prev.filter((c) => c.id !== id);
      const next = remaining.length > 0 ? remaining : [makeEmptyConversation()];
      if (id === activeConversationId) {
        setActiveConversationId(next[0].id);
      }
      return next;
    });
  };

  const handleAsk = async (question) => {
    const isFirstMessage = activeConversation.messages.length === 0;

    updateActiveConversation((c) => ({
      ...c,
      title: isFirstMessage ? question.slice(0, 48) : c.title,
      messages: [...c.messages, { id: makeId(), role: 'user', content: question }],
    }));

    setLoading(true);
    try {
      const result = await askQuestion(question);
      updateActiveConversation((c) => ({
        ...c,
        messages: [
          ...c.messages,
          {
            id: makeId(),
            role: 'assistant',
            content: result.answer,
            sources: result.sources,
          },
        ],
      }));
    } catch (err) {
      updateActiveConversation((c) => ({
        ...c,
        messages: [
          ...c.messages,
          {
            id: makeId(),
            role: 'assistant',
            content: `Sorry, couldn't reach the backend: ${err.message}`,
            sources: [],
          },
        ],
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleToggleSave = (message, question) => {
    setSaved((prev) => {
      const existingIndex = prev.findIndex((s) => s.sourceMessageId === message.id);
      if (existingIndex !== -1) {
        return prev.filter((_, i) => i !== existingIndex);
      }
      return [
        ...prev,
        {
          sourceMessageId: message.id,
          question,
          answer: message.content,
          sources: message.sources,
        },
      ];
    });
  };

  const isSaved = (message) => saved.some((s) => s.sourceMessageId === message.id);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>SDG&E Wildfire Mitigation Plan — Filing Assistant</h1>
        <span className={`status-dot ${backendOnline ? '' : 'offline'}`} />
        <span className="status-label">{backendOnline ? 'backend online' : 'backend unreachable'}</span>
      </header>

      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        saved={saved}
        onRemoveSaved={(i) => setSaved((prev) => prev.filter((_, idx) => idx !== i))}
      />

      <div className="chat-column">
        <div className="chat-scroll">
          {activeConversation.messages.length === 0 && (
            <p className="chat-empty">Ask a question below to get started.</p>
          )}
          {activeConversation.messages.map((m, i) => (
            <ChatMessage
              key={m.id}
              role={m.role}
              content={m.content}
              sources={m.sources}
              onToggleSave={
                m.role === 'assistant'
                  ? () => handleToggleSave(m, activeConversation.messages[i - 1]?.content)
                  : null
              }
              saved={m.role === 'assistant' ? isSaved(m) : false}
            />
          ))}
        </div>
        <ChatInput onSubmit={handleAsk} disabled={loading} />
      </div>
    </div>
  );
}

export default App;