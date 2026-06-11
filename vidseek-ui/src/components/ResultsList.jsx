import { ResultCard } from './ResultCard';

export function ResultsList({ results, query, loading, error, onPlay }) {
  if (loading) return (
    <div className="state-panel">
      <div className="state-glyph pulse">⬡</div>
      <p>Searching across all video content…</p>
    </div>
  );

  if (error) return (
    <div className="state-panel">
      <div className="state-glyph">⚠</div>
      <p>Could not reach the API.</p>
      <p className="state-sub">{error}</p>
    </div>
  );

  if (!query) return (
    <div className="state-panel">
      <div className="state-glyph">⌕</div>
      <p>Search across transcripts, detected objects,<br />on-screen text, and visual relationships.</p>
    </div>
  );

  if (!results.length) return (
    <div className="state-panel">
      <div className="state-glyph">∅</div>
      <p>No matches for <em>"{query}"</em></p>
    </div>
  );

  return (
    <div className="results-list">
      {results.map((r, i) => (
        <ResultCard key={i} result={r} index={i} onPlay={onPlay} />
      ))}
    </div>
  );
}
