import { useRef, useState } from 'react';
import { UploadCloud, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { useUpload } from '../hooks/useUpload';

const ACCEPT = '.mp4,.avi,.mov,.mkv,.webm';

const DETECTOR_OPTS   = [{ value: 'craft', label: 'CRAFT' }, { value: 'east', label: 'EAST' }];
const RECOGNIZER_OPTS = [{ value: 'easyocr', label: 'EasyOCR' }, { value: 'mser', label: 'MSER + HOG' }];
const OBJECT_DETECTOR_OPTS = [{ value: 'faster_rcnn', label: 'Faster R-CNN' }, { value: 'hog', label: 'HOG' }];

export function UploadPanel() {
  const fileRef = useRef(null);
  const { phase, uploadPct, message, startUpload, reset } = useUpload();
  const [detector,   setDetector]   = useState('craft');
  const [recognizer, setRecognizer] = useState('easyocr');
  const [object_detector, setObjectDetector] = useState('faster_rcnn');

  const handleFiles = files => {
    const file = files[0];
    if (file) startUpload(file, { detector, recognizer, object_detector });
  };

  const onDrop = e => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const isIdle       = phase === 'idle';
  const isUploading  = phase === 'uploading';
  const isProcessing = phase === 'processing';
  const isDone       = phase === 'done';
  const isError      = phase === 'error';
  const isActive     = isUploading || isProcessing;

  return (
    <div className="upload-panel">

      {(isIdle || isDone || isError) && (
        <>
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
            <UploadCloud size={24} className="upload-icon" />
            <p className="upload-cta"><strong>Drop a video</strong> or click to browse</p>
            <p className="upload-hint">mp4 · avi · mov · mkv · webm</p>
          </div>

          <div className="ocr-options">
            <div className="ocr-option-row">
              <span className="ocr-option-label">Detector</span>
              <div className="ocr-pills">
                {DETECTOR_OPTS.map(o => (
                  <button
                    key={o.value}
                    className={`ocr-pill${detector === o.value ? ' active' : ''}`}
                    onClick={() => setDetector(o.value)}
                  >{o.label}</button>
                ))}
              </div>
            </div>
            <div className="ocr-option-row">
              <span className="ocr-option-label">Recognizer</span>
              <div className="ocr-pills">
                {RECOGNIZER_OPTS.map(o => (
                  <button
                    key={o.value}
                    className={`ocr-pill${recognizer === o.value ? ' active' : ''}`}
                    onClick={() => setRecognizer(o.value)}
                  >{o.label}</button>
                ))}
              </div>
            </div>
            <div className="ocr-option-row">
              <span className="ocr-option-label">Object Detector</span>
              <div className="ocr-pills">
                {OBJECT_DETECTOR_OPTS.map(o => (
                  <button
                    key={o.value}
                    className={`ocr-pill${object_detector === o.value ? ' active' : ''}`}
                    onClick={() => setObjectDetector(o.value)}
                  >{o.label}</button>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {isActive && (
        <div className="upload-progress">
          <div className="progress-header">
            <Loader2 size={13} className="spin" />
            <span>{message}</span>
          </div>
          <div className="progress-track">
            {isUploading
              ? <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
              : <div className="progress-fill indeterminate" />
            }
          </div>
        </div>
      )}

      {isDone && (
        <div className="upload-done">
          <CheckCircle size={15} className="done-icon" />
          <span>Video processed - ready to search.</span>
          <button className="upload-reset" onClick={reset}>Upload another</button>
        </div>
      )}

      {isError && (
        <div className="upload-error">
          <XCircle size={15} className="error-icon" />
          <span>{message}</span>
          <button className="upload-reset" onClick={reset}>Try again</button>
        </div>
      )}

    </div>
  );
}
