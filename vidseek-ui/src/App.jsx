import { useState } from 'react';
import { Upload } from 'lucide-react';
import { SearchBar }         from './components/SearchBar';
import { SearchPanel }       from './components/SearchPanel';
import { ObjectSearchPanel } from './components/ObjectSearchPanel';
import { VRDSearchPanel }    from './components/VRDSearchPanel';
import { ResultsList }       from './components/ResultsList';
import { UploadPanel }       from './components/UploadPanel';
import { useSearch }         from './hooks/useSearch';

export default function App() {
  const { results, loading, error, query, search, changeSource } = useSearch();

  const [uploadOpen,  setUploadOpen]  = useState(false);
  const [activeMode,  setActiveMode]  = useState(null);    // 'text' | 'object' | 'vrd'
  const [sideResults, setSideResults] = useState([]);
  const [sideQuery,   setSideQuery]   = useState(null);
  const [sideLoading, setSideLoading] = useState(false);

  const activeResults = activeMode === 'text' ? results      : sideResults;
  const activeLoading = activeMode === 'text' ? loading      : sideLoading;
  const activeError   = activeMode === 'text' ? error        : null;
  const activeQuery   = activeMode === 'text' ? query        : sideQuery;

  const handleTextSearch = q => {
    if (q.trim()) setActiveMode('text');
    else if (activeMode === 'text') setActiveMode(null);
    search(q);
  };

  const handleSideResults = (mode, res, label) => {
    setActiveMode(mode);
    setSideResults(res);
    setSideQuery(label);
  };

  return (
    <div className="app">

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

      {/* Text search */}
      <div className="text-search-row">
        <div className="text-search-bar">
          <p className="section-eyebrow">Search</p>
          <SearchBar
            onSearch={handleTextSearch}
            loading={loading && activeMode === 'text'}
          />
          <SearchPanel onSourceChange={changeSource} />
        </div>
      </div>

      {/* Object + VRD panels */}
      <div className="panels-row">
        <ObjectSearchPanel
          onResults={(res, label) => handleSideResults('object', res, label)}
          onLoading={v => { setSideLoading(v); if (v) setActiveMode('object'); }}
        />
        <VRDSearchPanel
          onResults={(res, label) => handleSideResults('vrd', res, label)}
          onLoading={v => { setSideLoading(v); if (v) setActiveMode('vrd'); }}
        />
      </div>

      {/* Results — only shown once a search has run */}
      {(activeMode || activeLoading) && (
        <div className="results-section">
          <div className="results-section-header">
            <span className="section-eyebrow">Results</span>
            {activeMode && (
              <span className="mode-badge">{activeMode} search</span>
            )}
          </div>
          <ResultsList
            results={activeResults}
            query={activeQuery}
            loading={activeLoading}
            error={activeError}
          />
        </div>
      )}

    </div>
  );
}
