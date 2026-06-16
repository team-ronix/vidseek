import { useState, useCallback, useRef } from 'react';
import { uploadVideo, getJobStatus } from '../api/client';

const POLL_INTERVAL = 2500;

export function useUpload() {
  const [phase, setPhase]       = useState('idle');   // idle | uploading | processing | done | error
  const [uploadPct, setUploadPct] = useState(0);
  const [message, setMessage]   = useState('');
  const pollRef = useRef(null);

  const reset = () => {
    clearInterval(pollRef.current);
    setPhase('idle');
    setUploadPct(0);
    setMessage('');
  };

  const startUpload = useCallback(async (file) => {
    setPhase('uploading');
    setUploadPct(0);
    setMessage(`Uploading ${file.name}…`);

    try {
      const { job_id } = await uploadVideo(file, (pct) => setUploadPct(pct));

      setPhase('processing');
      setMessage('Pipeline starting…');

      pollRef.current = setInterval(async () => {
        try {
          const job = await getJobStatus(job_id);
          setMessage(job.message);
          if (job.status === 'done') {
            clearInterval(pollRef.current);
            setPhase('done');
          } else if (job.status === 'error') {
            clearInterval(pollRef.current);
            setPhase('error');
          }
        } catch {
          setMessage('Polling… retrying');
        }
      }, POLL_INTERVAL);
    } catch (e) {
      setPhase('error');
      setMessage(e.message);
    }
  }, []);

  return { phase, uploadPct, message, startUpload, reset };
}
