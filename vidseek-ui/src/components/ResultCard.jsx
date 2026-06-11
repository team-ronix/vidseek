import { Play } from 'lucide-react';

function fmtTime(secs) {
  if (secs == null || secs === 0) return null;
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

export function ResultCard({ result, index, onPlay }) {
  const start = fmtTime(result.start_time);
  const end   = fmtTime(result.end_time);
  const hasTime = start !== null;

  return (
    <div
      className="result-card"
      style={{ animationDelay: `${index * 30}ms` }}
      onClick={() => onPlay(result)}
    >
      <div className="card-body">
        <div className="card-meta">
          <span className={`type-badge type-${result.type}`}>{result.type}</span>
          {hasTime && (
            <span className="card-time">{start} → {end}</span>
          )}
        </div>
        <p className="card-text">{result.text}</p>
        <p className="card-path">{result.video_path}</p>
      </div>
      <button
        className="card-play"
        aria-label="Play at this moment"
        onClick={e => { e.stopPropagation(); onPlay(result); }}
      >
        <Play size={14} fill="currentColor" />
      </button>
    </div>
  );
}
