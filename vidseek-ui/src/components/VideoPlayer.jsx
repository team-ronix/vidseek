import { useRef } from 'react';
import { ArrowLeft, Clock, Play } from 'lucide-react';
import { getVideoStreamUrl } from '../api/client';

function fmtTime(secs) {
  if (secs == null) return null;
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

export function VideoPlayer({ video, onBack }) {
  const videoRef = useRef(null);

  const seekTo = (startTime) => {
    if (!videoRef.current) return;
  videoRef.current.currentTime = startTime;
    videoRef.current.play();
  };

  return (
    <div className="video-player-view">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={14} />
        <span>All videos</span>
      </button>

      <h3 className="video-player-title">{video.video_name}</h3>

      <video
        ref={videoRef}
        className="video-element"
        src={getVideoStreamUrl(video.video_path)}
        controls
        preload="metadata"
      />

      <div className="video-results-list">
        <p className="video-results-header">
          <strong>{video.match_count}</strong> match{video.match_count !== 1 ? 'es' : ''} - click any to seek
        </p>
        {video.results.map((r, i) => {
          const start = fmtTime(r.start_time);
          const end   = fmtTime(r.end_time);
          return (
            <button
              key={i}
              className="video-result-item"
              onClick={() => seekTo(r.frame_time || r.start_time)}
              title={`Seek to ${r.frame_time || r.start_time  }`}
            >
              <span className={`type-badge type-${r.type}`}>{r.type}</span>
              <span className="video-result-text">{r.text}</span>
              {start && (
                <span className="video-result-time">
                  <Clock size={11} />
                  {start}{end ? ` → ${end}` : ''}
                </span>
              )}
              <span className="video-result-play">
                <Play size={11} />
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
