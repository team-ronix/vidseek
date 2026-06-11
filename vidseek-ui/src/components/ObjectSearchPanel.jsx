import { useState, useEffect } from 'react';
import { Package, Loader2, AlertCircle } from 'lucide-react';
import { getAllObjects, searchByObject } from '../api/client';

export function ObjectSearchPanel({ onResults, onLoading }) {
  const [objects,  setObjects]  = useState([]);
  const [fetching, setFetching] = useState(true);
  const [fetchErr, setFetchErr] = useState(null);
  const [selected, setSelected] = useState('');
  const [searching, setSearching] = useState(false);

  // Load all objects on mount
  useEffect(() => {
    getAllObjects()
      .then(data => setObjects(data))
      .catch(e  => setFetchErr(e.message))
      .finally(() => setFetching(false));
  }, []);

  const handleSelect = async (key) => {
    setSelected(key);
    if (!key) { onResults([], null); return; }
    setSearching(true);
    onLoading(true);
    try {
      const results = await searchByObject(key);
      onResults(results, key);
    } catch (e) {
      onResults([], key);
    } finally {
      setSearching(false);
      onLoading(false);
    }
  };

  return (
    <div className="side-panel">
      <div className="side-panel-header">
        <Package size={15} className="side-panel-icon" />
        <h2 className="side-panel-title">Object Search</h2>
      </div>

      <p className="side-panel-desc">
        Find scenes where a specific object appears.
      </p>

      {fetching && (
        <div className="panel-loading">
          <Loader2 size={14} className="spin" />
          <span>Loading objects…</span>
        </div>
      )}

      {fetchErr && (
        <div className="panel-error">
          <AlertCircle size={13} />
          <span>{fetchErr}</span>
        </div>
      )}

      {!fetching && !fetchErr && (
        <>
          <div className="select-wrap">
            <select
              className="select"
              value={selected}
              onChange={e => handleSelect(e.target.value)}
              disabled={searching}
            >
              <option value="">Select Object</option>
              {objects.map(o => (
                <option key={o.id} value={o.key}>{o.key}</option>
              ))}
            </select>
            <span className="select-chevron">
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
                <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5"
                  strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </span>
          </div>

          {searching && (
            <div className="panel-loading" style={{ marginTop: 10 }}>
              <Loader2 size={13} className="spin" />
              <span>Searching…</span>
            </div>
          )}

          <div className="object-chips">
            {objects.slice(0, 12).map(o => (
              <button
                key={o.id}
                className={`object-chip${selected === o.key ? ' active' : ''}`}
                onClick={() => handleSelect(selected === o.key ? '' : o.key)}
              >
                {o.key}
              </button>
            ))}
            {objects.length > 12 && (
              <span className="chips-overflow">+{objects.length - 12} more in dropdown</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
