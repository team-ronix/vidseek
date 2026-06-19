import { useState } from 'react';
import { VideoCard }   from './VideoCard';
import { VideoPlayer } from './VideoPlayer';

export function ResultsList({ videos, query, loading, error }) {
  const [selectedVideo, setSelectedVideo] = useState(null);

  if (loading) return (
    <div className="state-panel">
      <div className="state-glyph pulse">⊙</div>
      <p>Searching…</p>
    </div>
  );

  if (error) return (
    <div className="state-panel">
      <div className="state-glyph">⚠</div>
      <p>Could not reach the API.</p>
      <p className="state-sub">{error}</p>
    </div>
  );

  if (!videos || !videos.length) return (
    <div className="state-panel">
      <div className="state-glyph">∅</div>
      <p>No matches found{query ? ` for "${query}"` : ''}.</p>
    </div>
  );

  if (selectedVideo) {
    return (
      <VideoPlayer
        video={selectedVideo}
        onBack={() => setSelectedVideo(null)}
      />
    );
  }

  const totalMatches = videos.reduce((sum, v) => sum + v.match_count, 0);

  return (
    <div className="results-list">
      <p className="results-count">
        <strong>{videos.length}</strong> video{videos.length !== 1 ? 's' : ''} &nbsp;·&nbsp;
        <strong>{totalMatches}</strong> total match{totalMatches !== 1 ? 'es' : ''}
      </p>
      {videos.map((v, i) => (
        <VideoCard
          key={v.video_path}
          video={v}
          index={i}
          onClick={setSelectedVideo}
        />
      ))}
    </div>
  );
}
