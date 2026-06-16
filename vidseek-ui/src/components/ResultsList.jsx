import { ResultCard } from './ResultCard';

export function ResultsList({ results, query, loading, error }) {
  if (loading) return (
    <div className="state-panel">
      <div className="state-glyph pulse">⊙</div>
      <p>Searching…</p>
    </div>
  );

  if (error) return (
    <div className="state-panel">
      <div className="state-glyph">⚠</div>
      <p>Could not reach the API.</p>
      <p className="state-sub">{error}</p>
    </div>
  );

  if (!results.length) return (
    <div className="state-panel">
      <div className="state-glyph">∅</div>
      <p>No matches found{query ? ` for "${query}"` : ''}.</p>
    </div>
  );

  return (
    <div className="results-list">
      <p className="results-count">
        <strong>{results.length}</strong> result{results.length !== 1 ? 's' : ''}
      </p>
      {results.map((r, i) => (
        <ResultCard key={`${r.type}-${i}`} result={r} index={i} />
      ))}
    </div>
  );
}
