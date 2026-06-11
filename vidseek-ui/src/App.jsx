import { useState, useMemo } from 'react';
import { Upload } from 'lucide-react';
import { SearchBar }   from './components/SearchBar';
import { FilterBar }   from './components/FilterBar';
import { ResultsList } from './components/ResultsList';
import { VideoModal }  from './components/VideoModal';
import { UploadPanel } from './components/UploadPanel';
import { useSearch }   from './hooks/useSearch';

export default function App() {
  const { results, loading, error, query, search } = useSearch();
  const [activeFilter, setActiveFilter]  = useState(null);
  const [activeResult, setActiveResult]  = useState(null);
  const [uploadOpen,   setUploadOpen]    = useState(false);

  const filtered = useMemo(
    () => activeFilter ? results.filter(r => r.type === activeFilter) : results,
    [results, activeFilter]
  );

  const handleFilter = type => {
    setActiveFilter(type);
  };

  return (
    <>
      <div className="app">
        {/* Header */}
        <header className="app-header">
          <div className="logo">
            <span className="logo-mark">vid<em>seek</em></span>
            <span className="logo-sub">semantic video search</span>
          </div>
          <button
            className={`upload-toggle${uploadOpen ? ' active' : ''}`}
            onClick={() => setUploadOpen(v => !v)}
          >
            <Upload size={14} />
            <span>{uploadOpen ? 'cancel' : 'upload video'}</span>
          </button>
        </header>

        {/* Upload panel */}
        {uploadOpen && <UploadPanel />}

        {/* Search */}
        <SearchBar onSearch={search} loading={loading} />

        {/* Stats + filters */}
        <div className="results-header">
          {query && !loading && !error && (
            <span className="results-count">
              <strong>{filtered.length}</strong> result{filtered.length !== 1 ? 's' : ''}{' '}
              {activeFilter ? `· ${activeFilter} only` : ''}
            </span>
          )}
          <FilterBar
            results={results}
            activeFilter={activeFilter}
            onFilter={handleFilter}
          />
        </div>

        {/* Results */}
        <ResultsList
          results={filtered}
          query={query}
          loading={loading}
          error={error}
          onPlay={setActiveResult}
        />
      </div>

      {/* Modal */}
      <VideoModal result={activeResult} onClose={() => setActiveResult(null)} />
    </>
  );
}
