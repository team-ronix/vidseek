import { useState } from 'react';

const SOURCES = [
  { id: 'transcript', label: 'Transcription' },
  { id: 'ocr',        label: 'On-screen Text' },
];

const MODELS = [
  { id: 'transformer', label: 'Transformer' },
  { id: 'hybrid',      label: 'HybridEmbedder' },
  { id: 'both',        label: 'Both' },
];

export function SearchPanel({ onSourceChange, onModelChange }) {
  const [activeSource, setActiveSource] = useState('transcript');
  const [activeModel,  setActiveModel]  = useState('transformer');

  const handleSource = (id) => { setActiveSource(id); onSourceChange(id); };
  const handleModel  = (id) => { setActiveModel(id);  onModelChange?.(id); };

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

      <div className="search-filter-row">
        <span className="search-panel-label">Model</span>
        <div className="source-pill-row">
          {MODELS.map(m => (
            <button
              key={m.id}
              className={`source-pill model-pill${activeModel === m.id ? ' active' : ''} pill-${m.id}`}
              onClick={() => handleModel(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
