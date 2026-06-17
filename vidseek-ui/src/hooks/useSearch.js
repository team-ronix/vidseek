import { useState, useCallback, useRef } from 'react';
import { searchVideos } from '../api/client';

export function useSearch() {
  const [rawResults,   setRawResults]   = useState([]);
  const [rawVideos,    setRawVideos]    = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [query,        setQuery]        = useState('');
  const [sourceFilter, setSourceFilter] = useState('all');
  const debounceRef = useRef(null);

  const search = useCallback(q => {
    setQuery(q);
    clearTimeout(debounceRef.current);
    if (!q.trim()) { setRawResults([]); setRawVideos([]); setError(null); return; }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await searchVideos(q);
        setRawResults(data.results || []);
        setRawVideos(data.videos || []);
      } catch (e) {
        setError(e.message);
        setRawResults([]);
        setRawVideos([]);
      } finally {
        setLoading(false);
      }
    }, 320);
  }, []);

  const changeSource = useCallback(src => setSourceFilter(src), []);

  const videos = sourceFilter === 'all'
    ? rawVideos
    : (() => {
        const filtered = rawResults.filter(r => r.type === sourceFilter);
        const groups = {};
        for (const r of filtered) {
          if (!groups[r.video_path]) {
            groups[r.video_path] = {
              video_path: r.video_path,
              video_name: r.video_path.split(/[\\/]/).pop(),
              match_count: 0,
              results: [],
            };
          }
          groups[r.video_path].match_count++;
          groups[r.video_path].results.push(r);
        }
        return Object.values(groups).sort((a, b) => b.match_count - a.match_count);
      })();

  return { videos, loading, error, query, search, changeSource };
}
