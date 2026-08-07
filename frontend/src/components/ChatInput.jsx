import { useState } from 'react';

function ChatInput({ onSubmit, disabled }) {
  const [value, setValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;
    onSubmit(value);
    setValue('');
  };

  return (
    <form className="chat-input-row" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask a question about SDG&E's Wildfire Mitigation Plan..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !value.trim()}>
        {disabled ? 'Asking…' : 'Ask'}
      </button>
    </form>
  );
}

export default ChatInput;