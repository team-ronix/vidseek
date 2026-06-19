import { FileVideo } from 'lucide-react';

export function VideoCard({ video, onClick, index }) {
  return (
    <button
      className="video-card"
      style={{ animationDelay: `${index * 30}ms` }}
      onClick={() => onClick(video)}
    >
      <div className="video-card-icon">
        <FileVideo size={22} />
      </div>
      <div className="video-card-info">
        <span className="video-card-name">{video.video_name}</span>
        <span className="video-card-path">{video.video_path}</span>
      </div>
      <span className="video-card-count">
        {video.match_count} match{video.match_count !== 1 ? 'es' : ''}
      </span>
    </button>
  );
}
