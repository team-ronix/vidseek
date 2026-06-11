import { useEffect, useRef, useState } from 'react';
import { X, ExternalLink } from 'lucide-react';
import { listVideos, videoStreamUrl } from '../api/client';

export function VideoModal({ result, onClose }) {
  const videoRef = useRef(null);
  const [streamUrl, setStreamUrl] = useState(null);
  const open = !!result;

  // Resolve video_id → stream URL whenever result changes
  useEffect(() => {
    if (!result) return;
    setStreamUrl(null);

    listVideos().then(videos => {
      const match = videos.find(
        v => v.file_path === result.video_path ||
             v.file_name === result.video_path.split('/').pop()
      );
      if (match) setStreamUrl(videoStreamUrl(match.id));
    }).catch(() => {});
  }, [result]);

  // Seek to start_time once metadata loads
  useEffect(() => {
    if (!streamUrl || !videoRef.current) return;
    const el = videoRef.current;
    const seek = () => {
      if (result?.start_time > 0) el.currentTime = result.start_time;
    };
    el.addEventListener('loadedmetadata', seek);
    return () => el.removeEventListener('loadedmetadata', seek);
  }, [streamUrl, result]);

  // Close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // Pause + reset on close
  useEffect(() => {
    if (!open && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
    }
  }, [open]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" role="dialog" aria-modal="true">

        <div className="modal-header">
          <span className={`type-badge type-${result.type}`}>{result.type}</span>
          <span className="modal-title">{result.text}</span>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="modal-video-wrap">
          {streamUrl
            ? <video ref={videoRef} src={streamUrl} controls autoPlay className="modal-video" />
            : <div className="modal-video-placeholder">
                <div className="state-glyph pulse">⬡</div>
                <p>Resolving video…</p>
              </div>
          }
        </div>

        <div className="modal-footer">
          <span className="modal-path">{result.video_path}</span>
          {result.start_time > 0 && (
            <span className="modal-timestamp">
              @ {Math.floor(result.start_time / 60)}:{String(Math.floor(result.start_time % 60)).padStart(2,'0')}
            </span>
          )}
        </div>

      </div>
    </div>
  );
}
