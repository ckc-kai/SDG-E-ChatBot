function SavedQueriesSidebar({ saved, onRemove }) {
    return (
      <aside className="sidebar">
        <h2>Saved Queries</h2>
        {saved.length === 0 && (
          <p className="sidebar-empty">Nothing saved yet — star an answer to keep it here.</p>
        )}
        {saved.map((item, i) => (
          <div className="saved-item" key={i}>
            <p className="saved-item-question">{item.question}</p>
            <button className="saved-item-remove" onClick={() => onRemove(i)}>
              Remove
            </button>
          </div>
        ))}
      </aside>
    );
  }
  
  export default SavedQueriesSidebar;