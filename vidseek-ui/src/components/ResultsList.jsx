import { useState } from 'react';
import { VideoCard }   from './VideoCard';
import { VideoPlayer } from './VideoPlayer';

function LatencyBar({ latency }) {
  const entries = Object.entries(latency || {});
  if (!entries.length) return null;
  return (
    <div className="latency-bar">
      {entries.map(([model, ms]) => (
        <span key={model} className={`latency-chip latency-${model}`}>
          {model} <strong>{ms} ms</strong>
        </span>
      ))}
    </div>
  );
}

export function ResultsList({ videos, query, loading, error, latency }) {
  const [selectedVideo, setSelectedVideo] = useState(null);

  if (loading) return (
    <div className="state-panel">
      <div className="state-glyph pulse">&#x2299;</div>
      <p>Searching&hellip;</p>
    </div>
  );

  if (error) return (
    <div className="state-panel">
      <div className="state-glyph">&#x26A0;</div>
      <p>Could not reach the API.</p>
      <p className="state-sub">{error}</p>
    </div>
  );

  if (!videos || !videos.length) return (
    <>
      <LatencyBar latency={latency} />
      <div className="state-panel">
        <div className="state-glyph">&#x2205;</div>
        <p>No matches found{query ? ` for "${query}"` : ''}.</p>
      </div>
    </>
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
      <LatencyBar latency={latency} />
      <p className="results-count">
        <strong>{videos.length}</strong> video{videos.length !== 1 ? 's' : ''} &nbsp;&middot;&nbsp;
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
