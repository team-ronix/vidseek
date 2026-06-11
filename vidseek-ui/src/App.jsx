import { useState } from 'react';
import { Upload } from 'lucide-react';
import { SearchBar }        from './components/SearchBar';
import { SearchPanel }      from './components/SearchPanel';
import { ObjectSearchPanel } from './components/ObjectSearchPanel';
import { VRDSearchPanel }   from './components/VRDSearchPanel';
import { ResultsList }      from './components/ResultsList';
import { UploadPanel }      from './components/UploadPanel';
import { useSearch }        from './hooks/useSearch';

// Which search mode produced the current results
const MODE_TEXT   = 'text';
const MODE_OBJECT = 'object';
const MODE_VRD    = 'vrd';

export default function App() {
  const { results: textResults, loading: textLoading, error: textError,
          query, search, changeSource } = useSearch();

  const [uploadOpen,     setUploadOpen]     = useState(false);
  const [mode,           setMode]           = useState(null);   // text | object | vrd
  const [sideResults,    setSideResults]    = useState([]);
  const [sideQuery,      setSideQuery]      = useState(null);
  const [sideLoading,    setSideLoading]    = useState(false);

  // Determine what to show in results section
  const activeResults = mode === MODE_TEXT   ? textResults
                      : mode === MODE_OBJECT || mode === MODE_VRD ? sideResults
                      : [];
  const activeLoading = mode === MODE_TEXT   ? textLoading  : sideLoading;
  const activeError   = mode === MODE_TEXT   ? textError    : null;
  const activeQuery   = mode === MODE_TEXT   ? query        : sideQuery;

  const handleTextSearch = (q) => {
    if (q.trim()) setMode(MODE_TEXT);
    else if (mode === MODE_TEXT) setMode(null);
    search(q);
  };

  const handleObjectResults = (res, label) => {
    setMode(MODE_OBJECT);
    setSideResults(res);
    setSideQuery(label);
  };

  const handleVRDResults = (res, label) => {
    setMode(MODE_VRD);
    setSideResults(res);
    setSideQuery(label);
  };

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-mark">vid<em>seek</em></span>
          <span className="logo-sep">/</span>
          <span className="logo-sub">semantic video search</span>
        </div>
        <button
          className={`upload-toggle${uploadOpen ? ' active' : ''}`}
          onClick={() => setUploadOpen(v => !v)}
        >
          <Upload size={13} />
          <span>{uploadOpen ? 'cancel' : 'upload video'}</span>
        </button>
      </header>

      {uploadOpen && <UploadPanel />}

      {/* ── Text search row ─────────────────────────────────── */}
      <div className="text-search-row">
        <div className="text-search-bar">
          <p className="section-eyebrow">Search</p>
          <SearchBar onSearch={handleTextSearch} loading={textLoading && mode === MODE_TEXT} />
          <SearchPanel onSourceChange={changeSource} />
        </div>
      </div>

      {/* ── Two side-by-side panels ─────────────────────────── */}
      <div className="panels-row">
        <ObjectSearchPanel
          onResults={handleObjectResults}
          onLoading={v => { setSideLoading(v); if (v) setMode(MODE_OBJECT); }}
        />
        <VRDSearchPanel
          onResults={handleVRDResults}
          onLoading={v => { setSideLoading(v); if (v) setMode(MODE_VRD); }}
        />
      </div>

      {/* ── Results ─────────────────────────────────────────── */}
      {(mode || activeLoading) && (
        <div className="results-section">
          <div className="results-section-header">
            <span className="section-eyebrow">Results</span>
            {mode && (
              <span className="mode-badge mode-badge--{mode}">{mode} search</span>
            )}
          </div>
          <ResultsList
            results={activeResults}
            query={activeQuery}
            loading={activeLoading}
            error={activeError}
            source={mode}
          />
        </div>
      )}

    </div>
  );
}
