import { useState, useEffect } from 'react';
import { GitMerge, Loader2, AlertCircle, RotateCcw } from 'lucide-react';
import { getAllVRDOptions, searchByVRD } from '../api/client';

function VRDSelect({ label, placeholder, value, options, onChange, disabled }) {
  return (
    <div className="vrd-field">
      <label className="field-label">{label}</label>
      <div className="select-wrap">
        <select
          className="select"
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
        >
          <option value="">{placeholder}</option>
          {options.map(o => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
        <span className="select-chevron">
          <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
            <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </span>
      </div>
    </div>
  );
}

export function VRDSearchPanel({ onResults, onLoading }) {
  const [options,  setOptions]  = useState({ subjects: [], relations: [], objects: [] });
  const [fetching, setFetching] = useState(true);
  const [fetchErr, setFetchErr] = useState(null);

  const [subject,  setSubject]  = useState('');
  const [relation, setRelation] = useState('');
  const [object,   setObject]   = useState('');
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    getAllVRDOptions()
      .then(data => setOptions(data))
      .catch(e   => setFetchErr(e.message))
      .finally(() => setFetching(false));
  }, []);

  const hasSelection = subject || relation || object;

  const runSearch = async (s, r, o) => {
    if (!s && !r && !o) { onResults([], null); return; }
    setSearching(true);
    onLoading(true);
    try {
      const data = await searchByVRD({ subject: s, relation: r, object: o });
      onResults(data.videos || [], `${s || '?'} - ${r || '?'} - ${o || '?'}`);
    } catch (e) {
      onResults([], null);
    } finally {
      setSearching(false);
      onLoading(false);
    }
  };

  const handleSubject  = v => { setSubject(v);  runSearch(v, relation, object); };
  const handleRelation = v => { setRelation(v); runSearch(subject, v, object); };
  const handleObject   = v => { setObject(v);   runSearch(subject, relation, v); };

  const handleReset = () => {
    setSubject(''); setRelation(''); setObject('');
    onResults([], null);
  };

  return (
    <div className="side-panel">
      <div className="side-panel-header">
        <GitMerge size={15} className="side-panel-icon" />
        <h2 className="side-panel-title">Relationship Search</h2>
        {hasSelection && (
          <button className="panel-reset-btn" onClick={handleReset} title="Clear filters">
            <RotateCcw size={12} />
          </button>
        )}
      </div>

      <p className="side-panel-desc">
        Find scenes by visual relationships between objects.
      </p>

      {fetching && (
        <div className="panel-loading">
          <Loader2 size={14} className="spin" />
          <span>Loading options…</span>
        </div>
      )}

      {fetchErr && (
        <div className="panel-error">
          <AlertCircle size={13} />
          <span>{fetchErr}</span>
        </div>
      )}

      {!fetching && !fetchErr && (
        <div className="vrd-fields">

          {/* Visual triple hint */}
          <div className="vrd-triple-hint">
            <span className={`triple-slot${subject  ? ' filled' : ''}`}>{subject  || 'Subject'}</span>
            <span className="triple-sep">-></span>
            <span className={`triple-slot${relation ? ' filled' : ''}`}>{relation || 'Relation'}</span>
            <span className="triple-sep">-></span>
            <span className={`triple-slot${object   ? ' filled' : ''}`}>{object   || 'Object'}</span>
          </div>

          <VRDSelect
            label="Subject"
            placeholder="Any subject"
            value={subject}
            options={options.subjects}
            onChange={handleSubject}
            disabled={searching}
          />
          <VRDSelect
            label="Relationship"
            placeholder="Any relationship"
            value={relation}
            options={options.relations}
            onChange={handleRelation}
            disabled={searching}
          />
          <VRDSelect
            label="Object"
            placeholder="Any object"
            value={object}
            options={options.objects}
            onChange={handleObject}
            disabled={searching}
          />

          {searching && (
            <div className="panel-loading">
              <Loader2 size={13} className="spin" />
              <span>Searching…</span>
            </div>
          )}

        </div>
      )}
    </div>
  );
}
