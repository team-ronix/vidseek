const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function getVideoStreamUrl(path) {
  return `${BASE}/video/stream?path=${encodeURIComponent(path)}`;
}

async function request(url) {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`Request failed: ${res.status} ${url}`);
  return res.json();
}

// â”€â”€ Text search (ChromaDB: OCR + transcript) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export async function searchVideos(query, topK = 20) {
  return request(`/search?q=${encodeURIComponent(query)}&top_k=${topK}`);
}

export async function searchByOCR(query) {
  return request(`/search/ocr?q=${encodeURIComponent(query)}`);
}

// â”€â”€ Structured search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export async function searchByObject(objectKey) {
  return request(`/search/object?key=${encodeURIComponent(objectKey)}`);
}

export async function searchByVRD({ subject, object, relation }) {
  const params = new URLSearchParams();
  if (subject)  params.set('subject',  subject);
  if (object)   params.set('object',   object);
  if (relation) params.set('relation', relation);
  return request(`/search/vrd?${params}`);
}

// â”€â”€ Dropdown population â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export async function getAllObjects() {
  return request('/objects');            // [{ id, key }]
}

export async function getAllVRDOptions() {
  return request('/vrd/options');        // { subjects, relations, objects }
}

// â”€â”€ Upload + job polling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export async function getJobStatus(jobId) {
  return request(`/jobs/${jobId}`);
}

export function uploadVideo(file, onProgress, options = {}) {
  return new Promise((resolve, reject) => {
    const xhr  = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);
    if (options.detector)   form.append('detector',   options.detector);
    if (options.recognizer) form.append('recognizer', options.recognizer);
    if (options.object_detector) form.append('object_detector', options.object_detector);
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
    });
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.status}`));
    });
    xhr.addEventListener('error', () => reject(new Error('Network error')));
    xhr.open('POST', `${BASE}/videos/upload`);
    xhr.send(form);
  });
}

// -- Chapters ------------------------------------------
export async function getChapters(videoPath) {
  return request(`/videos/chapters?path=${encodeURIComponent(videoPath)}`);
}

