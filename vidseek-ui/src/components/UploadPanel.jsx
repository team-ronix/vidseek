import { useRef } from 'react';
import { UploadCloud, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { useUpload } from '../hooks/useUpload';

const ACCEPT = '.mp4,.avi,.mov,.mkv,.webm';

export function UploadPanel() {
  const fileRef = useRef(null);
  const { phase, uploadPct, message, startUpload, reset } = useUpload();

  const handleFiles = files => {
    const file = files[0];
    if (!file) return;
    startUpload(file);
  };

  const onDrop = e => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const isDone       = phase === 'done';
  const isError      = phase === 'error';
  const isUploading  = phase === 'uploading';
  const isProcessing = phase === 'processing';
  const isIdle       = phase === 'idle';
  const isActive     = isUploading || isProcessing;

  return (
    <div className="upload-panel">

      {/* Drop zone — hidden while active */}
      {(isIdle || isDone || isError) && (
        <div
          className="upload-drop"
          onDragOver={e => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={e => handleFiles(e.target.files)}
          />
          <UploadCloud size={28} className="upload-icon" />
          <p className="upload-cta"><strong>Drop a video</strong> or click to browse</p>
          <p className="upload-hint">mp4 · avi · mov · mkv · webm</p>
        </div>
      )}

      {/* Progress */}
      {isActive && (
        <div className="upload-progress">
          <div className="progress-header">
            <Loader2 size={14} className="spin" />
            <span>{message}</span>
          </div>
          {isUploading && (
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
            </div>
          )}
          {isProcessing && (
            <div className="progress-track">
              <div className="progress-fill indeterminate" />
            </div>
          )}
        </div>
      )}

      {/* Done */}
      {isDone && (
        <div className="upload-done">
          <CheckCircle size={16} className="done-icon" />
          <span>Video processed — ready to search.</span>
          <button className="upload-reset" onClick={reset}>Upload another</button>
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="upload-error">
          <XCircle size={16} className="error-icon" />
          <span>{message}</span>
          <button className="upload-reset" onClick={reset}>Try again</button>
        </div>
      )}

    </div>
  );
}
