import { FileVideo, Clock } from 'lucide-react';

function fmtTime(secs) {
  if (secs == null || secs === 0) return null;
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

export function ResultCard({ result, index }) {
  const start = fmtTime(result.start_time);
  const end   = fmtTime(result.end_time);

  return (
    <div className="result-card" style={{ animationDelay: `${index * 25}ms` }}>
      <div className="result-card-inner">

        {/* Type badge */}
        <span className={`type-badge type-${result.type}`}>{result.type}</span>

        {/* Model badge – only shown when comparing both models */}
        {result.source_model && result.source_model !== 'transformer' && (
          <span className={`model-badge model-${result.source_model}`}>{result.source_model}</span>
        )}

        {/* Matched text / triple */}
        <p className="card-text">{result.text}</p>

        {/* Location info */}
        <div className="card-location">
          <span className="card-loc-item">
            <FileVideo size={12} />
            <span className="card-path">{result.video_path}</span>
          </span>
          {start && (
            <span className="card-loc-item card-timestamp">
              <Clock size={12} />
              <span>{start}{end ? ` → ${end}` : ''}</span>
            </span>
          )}
        </div>

      </div>
    </div>
  );
}
