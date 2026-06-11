const TYPES = ['ocr', 'transcript', 'object', 'vrd'];

export function FilterBar({ results, activeFilter, onFilter }) {
  if (!results.length) return null;

  const counts = TYPES.reduce((acc, t) => {
    acc[t] = results.filter(r => r.type === t).length;
    return acc;
  }, {});

  return (
    <div className="filter-bar">
      <span className="filter-label">filter</span>
      {TYPES.map(t => (
        counts[t] > 0 && (
          <button
            key={t}
            className={`filter-chip type-${t}${activeFilter === t ? ' active' : ''}`}
            onClick={() => onFilter(activeFilter === t ? null : t)}
          >
            {t} <span className="chip-count">{counts[t]}</span>
          </button>
        )
      ))}
    </div>
  );
}
