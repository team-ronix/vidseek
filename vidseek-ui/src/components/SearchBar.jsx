import { useRef } from 'react';
import { Search, Loader2 } from 'lucide-react';

export function SearchBar({ onSearch, loading }) {
  const inputRef = useRef(null);

  return (
    <div className="search-wrap">
      <span className="search-adornment">
        {loading
          ? <Loader2 size={16} className="spin" />
          : <Search size={16} />}
      </span>
      <input
        ref={inputRef}
        className="search-input"
        type="text"
        placeholder="Search for video content..."
        autoComplete="off"
        spellCheck="false"
        onChange={e => onSearch(e.target.value)}
      />
    </div>
  );
}
