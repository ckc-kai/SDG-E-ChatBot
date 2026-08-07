import { useEffect, useState } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import SavedQueriesSidebar from './components/SavedQueriesSidebar';
import { askQuestion, checkHealth } from './api/client';

function App() {
  const [messages, setMessages] = useState([]); // [{role, content, sources?}]
  const [saved, setSaved] = useState([]);        // [{question, answer, sources}]
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);

  useEffect(() => {
    checkHealth().then(setBackendOnline).catch(() => setBackendOnline(false));
  }, []);

  const handleAsk = async (question) => {
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);
    try {
      const result = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: result.answer, sources: result.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, couldn't reach the backend: ${err.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = (message, question) => {
    setSaved((prev) => [...prev, { question, answer: message.content, sources: message.sources }]);
  };

  const isSaved = (message) => saved.some((s) => s.answer === message.content);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>SDG&E Wildfire Mitigation Plan — Filing Assistant</h1>
        <span className={`status-dot ${backendOnline ? '' : 'offline'}`} />
        <span className="status-label">{backendOnline ? 'backend online' : 'backend unreachable'}</span>
      </header>

      <SavedQueriesSidebar
        saved={saved}
        onRemove={(i) => setSaved((prev) => prev.filter((_, idx) => idx !== i))}
      />

      <div className="chat-column">
        <div className="chat-scroll">
          {messages.length === 0 && (
            <p className="chat-empty">Ask a question below to get started.</p>
          )}
          {messages.map((m, i) => (
            <ChatMessage
              key={i}
              role={m.role}
              content={m.content}
              sources={m.sources}
              onSave={m.role === 'assistant' ? () => handleSave(m, messages[i - 1]?.content) : null}
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