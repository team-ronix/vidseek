import { useState, useCallback, useRef } from 'react';
import { searchVideos } from '../api/client';

export function useSearch() {
  const [results,      setResults]      = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [query,        setQuery]        = useState('');
  const [sourceFilter, setSourceFilter] = useState('all');
  const debounceRef = useRef(null);

  const search = useCallback((q) => {
    setQuery(q);
    clearTimeout(debounceRef.current);
    if (!q.trim()) { setResults([]); setError(null); return; }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await searchVideos(q);
        setResults(data.results);
      } catch (e) {
        setError(e.message);
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 320);
  }, []);

  const changeSource = useCallback((src) => {
    setSourceFilter(src);
  }, []);

  const filtered = sourceFilter === 'all'
    ? results
    : results.filter(r => r.type === sourceFilter);

  return { results: filtered, loading, error, query, search, changeSource };
}
