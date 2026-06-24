import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, BookOpen, Clock, Play } from 'lucide-react';
import { getChapters, getVideoStreamUrl } from '../api/client';

function fmtTime(secs) {
  if (secs == null) return null;
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

export function VideoPlayer({ video, onBack }) {
  const videoRef = useRef(null);
  const [chapters, setChapters] = useState([]);
  const [activeTab, setActiveTab] = useState('matches'); // 'matches' | 'chapters'

  useEffect(() => {
    setChapters([]);
    getChapters(video.video_path)
      .then(setChapters)
      .catch(() => {});
  }, [video.video_path]);

  const seekTo = (time) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = time;
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

      <div className="video-tabs">
        <button
          className={`video-tab${activeTab === 'matches' ? ' active' : ''}`}
          onClick={() => setActiveTab('matches')}
        >
          <Play size={12} />
          <span>{video.match_count} match{video.match_count !== 1 ? 'es' : ''}</span>
        </button>
        {chapters.length > 0 && (
          <button
            className={`video-tab${activeTab === 'chapters' ? ' active' : ''}`}
            onClick={() => setActiveTab('chapters')}
          >
            <BookOpen size={12} />
            <span>{chapters.length} chapter{chapters.length !== 1 ? 's' : ''}</span>
          </button>
        )}
      </div>

      {activeTab === 'matches' && (
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
              >
                <span className={`type-badge type-${r.type}`}>{r.type}</span>
                <span className={`model-badge model-${r.source_model || "transformer"}`}>{r.source_model || "tf"}</span>
                <span className="video-result-text">{r.text}</span>
                {start && (
                  <span className="video-result-time">
                    <Clock size={11} />
                    {start}{end ? ` -> ${end}` : ''}
                  </span>
                )}
                <span className="video-result-play">
                  <Play size={11} />
                </span>
              </button>
            );
          })}
        </div>
      )}

      {activeTab === 'chapters' && (
        <div className="video-results-list">
          <p className="video-results-header">
            <strong>{chapters.length}</strong> chapter{chapters.length !== 1 ? 's' : ''} - click any to jump
          </p>
          {chapters.map((ch, i) => (
            <button
              key={ch.id}
              className="video-result-item chapter-item"
              onClick={() => seekTo(ch.start)}
            >
              <span className="chapter-index">{i + 1}</span>
              <span className="video-result-text chapter-title">{ch.title}</span>
              <span className="video-result-time">
                <Clock size={11} />
                {fmtTime(ch.start)}{ch.end ? ` -> ${fmtTime(ch.end)}` : ''}
              </span>
              <span className="video-result-play">
                <Play size={11} />
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

