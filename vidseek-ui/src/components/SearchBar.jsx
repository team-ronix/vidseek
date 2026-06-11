import { useRef } from 'react';
import { Search, Loader2 } from 'lucide-react';

export function SearchBar({ onSearch, loading }) {
  const inputRef = useRef(null);

  return (
    <div className="search-wrap">
      <input
        ref={inputRef}
        className="search-input"
        type="text"
        placeholder="search transcripts, objects, text on screen, visual relationships…"
        autoComplete="off"
        spellCheck="false"
        onChange={e => onSearch(e.target.value)}
      />
      <span className="search-adornment">
        {loading
          ? <Loader2 size={18} className="spin" />
          : <Search size={18} />}
      </span>
    </div>
  );
}
