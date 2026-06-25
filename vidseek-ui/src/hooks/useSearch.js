import { useState, useCallback, useRef, useEffect } from 'react';
import { searchVideos, searchByOCR } from '../api/client';

export function useSearch() {
  const [rawResults,   setRawResults]   = useState([]);
  const [rawVideos,    setRawVideos]    = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [query,        setQuery]        = useState('');
  const [sourceFilter, setSourceFilter] = useState('transcript');
  const [model,        setModel]        = useState('transformer');
  const [latency,      setLatency]      = useState({});
  const debounceRef = useRef(null);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    if (!query.trim()) {
      setRawResults([]);
      setRawVideos([]);
      setError(null);
      setLatency({});
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        if (sourceFilter === 'ocr') {
          const data = await searchByOCR(query);
          setRawResults(data || []);
          setRawVideos([]);
          setLatency({});
        } else {
          const data = await searchVideos(query, 20, model);
          setRawResults(data.results || []);
          setRawVideos(data.videos  || []);
          setLatency(data.latency_ms || {});
        }
      } catch (e) {
        setError(e.message);
        setRawResults([]);
        setRawVideos([]);
        setLatency({});
      } finally {
        setLoading(false);
      }
    }, 320);

    return () => clearTimeout(debounceRef.current);
  }, [query, sourceFilter, model]);

  const search      = useCallback(q   => setQuery(q),        []);
  const changeSource = useCallback(src => setSourceFilter(src), []);
  const changeModel  = useCallback(m   => setModel(m),         []);

  const videos = sourceFilter === 'ocr'
    ? (() => {
        const groups = {};
        for (const r of rawResults) {
          const key = r.video_path || '';
          if (!groups[key]) {
            groups[key] = {
              video_path: key,
              video_name: key.split(/[\\/]/).pop(),
              match_count: 0,
              results: [],
            };
          }
          groups[key].match_count++;
          groups[key].results.push(r);
        }
        return Object.values(groups).sort((a, b) => b.match_count - a.match_count);
      })()
    : rawVideos;

  return { videos, loading, error, query, search, changeSource, model, changeModel, latency };
}
