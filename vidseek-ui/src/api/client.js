const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function searchVideos(query, topK = 20) {
  const res = await fetch(`${BASE}/search?q=${encodeURIComponent(query)}&top_k=${topK}`);
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function listVideos() {
  const res = await fetch(`${BASE}/videos`);
  if (!res.ok) throw new Error(`Failed to list videos: ${res.status}`);
  return res.json();
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job not found: ${res.status}`);
  return res.json();
}

export async function getAllObjects() {
  const res = await fetch(`${BASE}/objects`);
  if (!res.ok) throw new Error(`Failed to fetch objects: ${res.status}`);
  return res.json(); // [{ id, key }]
}

export async function getAllVRDOptions() {
  const res = await fetch(`${BASE}/vrd/options`);
  if (!res.ok) throw new Error(`Failed to fetch VRD options: ${res.status}`);
  return res.json(); // { subjects: [], objects: [], relations: [] }
}

export async function searchByObject(objectKey) {
  const res = await fetch(`${BASE}/search/object?key=${encodeURIComponent(objectKey)}`);
  if (!res.ok) throw new Error(`Object search failed: ${res.status}`);
  return res.json();
}

export async function searchByVRD({ subject, object, relation }) {
  const params = new URLSearchParams();
  if (subject)  params.set('subject',  subject);
  if (object)   params.set('object',   object);
  if (relation) params.set('relation', relation);
  const res = await fetch(`${BASE}/search/vrd?${params}`);
  if (!res.ok) throw new Error(`VRD search failed: ${res.status}`);
  return res.json();
}

export function uploadVideo(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr  = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);
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
