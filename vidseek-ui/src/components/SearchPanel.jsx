import { useState } from 'react';

const SOURCES = [
  { id: 'all',        label: 'All Sources' },
  { id: 'transcript', label: 'Voice Transcription Only' },
  { id: 'ocr',        label: 'On-screen Text (OCR) Only' },
];

export function SearchPanel({ onSourceChange }) {
  const [active, setActive] = useState('all');

  const handle = (id) => {
    setActive(id);
    onSourceChange(id);
  };

  return (
    <div className="search-panel-inline">
      <span className="search-panel-label">Source</span>
      <div className="source-pill-row">
        {SOURCES.map(s => (
          <button
            key={s.id}
            className={`source-pill${active === s.id ? ' active' : ''}`}
            onClick={() => handle(s.id)}
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}
