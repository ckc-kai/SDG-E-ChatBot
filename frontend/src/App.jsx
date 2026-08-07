import { useEffect, useState } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import SavedQueriesSidebar from './components/SavedQueriesSidebar';
import { askQuestion, checkHealth, warmupModels } from './api/client';

function App() {
  const [messages, setMessages] = useState([]);
  const [saved, setSaved] = useState([]);
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);
  const [modelsReady, setModelsReady] = useState(false);

  useEffect(() => {
    const prepareBackend = async () => {
      try {
        const online = await checkHealth();
        setBackendOnline(online);
        if (online) {
          setModelsReady(await warmupModels());
        }
      } catch {
        setBackendOnline(false);
        setModelsReady(false);
      }
    };
    prepareBackend();
  }, []);

  const handleAsk = async (question) => {
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);
    try {
      const result = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: result.answer,
          citations: result.citations,
          insufficientContext: result.insufficient_context,
          requestId: result.request_id,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `The request could not be completed: ${err.message}`,
          citations: [],
          insufficientContext: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = (message, question) => {
    setSaved((prev) => [
      ...prev,
      { question, answer: message.content, citations: message.citations },
    ]);
  };

  const isSaved = (message) => saved.some((s) => s.answer === message.content);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>SDG&E Wildfire Mitigation Plan — Filing Assistant</h1>
        <span className={`status-dot ${backendOnline ? '' : 'offline'}`} />
        <span className="status-label">
          {!backendOnline
            ? 'backend unreachable'
            : modelsReady
              ? 'models ready'
              : 'preparing models...'}
        </span>
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
              citations={m.citations}
              insufficientContext={m.insufficientContext}
              requestId={m.requestId}
              onSave={m.role === 'assistant' ? () => handleSave(m, messages[i - 1]?.content) : null}
              saved={m.role === 'assistant' ? isSaved(m) : false}
            />
          ))}
        </div>
        <ChatInput onSubmit={handleAsk} disabled={loading || !modelsReady} />
      </div>
    </div>
  );
}

export default App;
