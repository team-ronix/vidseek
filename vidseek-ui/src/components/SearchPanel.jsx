import { useState } from 'react';

const SOURCES = [
  { id: 'transcript', label: 'Transcription' },
  { id: 'ocr',        label: 'On-screen Text' },
];

export function SearchPanel({ onSourceChange }) {
  const [activeSource, setActiveSource] = useState('transcript');

  const handleSource = (id) => { setActiveSource(id); onSourceChange(id); };

  return (
    <div className="search-panel-inline">
      <div className="search-filter-row">
        <span className="search-panel-label">Source</span>
        <div className="source-pill-row">
          {SOURCES.map(s => (
            <button
              key={s.id}
              className={`source-pill${activeSource === s.id ? ' active' : ''}`}
              onClick={() => handleSource(s.id)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
