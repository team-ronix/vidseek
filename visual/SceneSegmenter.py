import cv2
import math
import numpy as np
from datastructures import Scene
_area_sums_cache = {}

def _area_sums(parent_size, num_parts):
    key = (parent_size, num_parts)
    if key in _area_sums_cache:
        return _area_sums_cache[key]
    if num_parts == 0:
        result = frozenset({0})
    elif num_parts == 1:
        result = frozenset({parent_size * parent_size})
    elif parent_size < num_parts:
        result = frozenset()
    else:
        res = set()
        for i in range(int(round(parent_size / 2))):
            sq = (i+1) * (i+1)
            for s in _area_sums(parent_size-i-1, num_parts-1):
                res.add(sq+s)
        result = frozenset(res)
    _area_sums_cache[key] = result
    return result


def _bgr_to_hsv(img):
    bgr = img.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    rgb = np.stack([r, g, b], axis=-1)
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax-cmin
    h = np.zeros_like(cmax)
    mask_r = (cmax==r) & (delta!=0)
    mask_g = (cmax==g) & (delta!=0)
    mask_b = (cmax==b) & (delta!=0)
    h[mask_r] = 60.0* (((g[mask_r]-b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 *(((b[mask_g] -r[mask_g])/delta[mask_g])+ 2)
    h[mask_b] = 60.0 * (((r[mask_b] -g[mask_b])/delta[mask_b])+ 4)
    s = delta/np.maximum(cmax, 1e-12)
    h /= 2
    s *= 255
    cmax *= 255
    return np.stack([h.astype(np.uint8), s.astype(np.uint8), cmax.astype(np.uint8)], axis=-1)

def _calc_hist(channel, n_bins, value_range):
    return np.histogram(channel.ravel(),bins=n_bins,range=value_range)[0].astype(np.float32)


def _prefix_sum(dist_mat):
    N = len(dist_mat)
    ps = np.zeros((N+1, N+1))
    for i in range(1, N+1):
        for j in range(1, N+1):
            ps[i][j] = dist_mat[i-1][j-1]+ps[i-1][j]+ps[i][j-1]-ps[i-1][j-1]
    return ps


def detectShots(video_path, thresh=30, min_len=10):
    """Detect shot boundaries using HSV mean difference"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_hsv = None
    cuts = [0]
    fi = 0
    while True:
        
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if prev_hsv is not None:

            diff = np.abs(hsv.astype(np.int32)-prev_hsv.astype(np.int32)).mean()

            if diff > thresh:
                cuts.append(fi)
        prev_hsv = hsv
        fi += 1
    cap.release()
    cuts.append(fi)
    shots = [(cuts[i], cuts[i+1]-1) for i in range(len(cuts)-1)]
    shots = [s for s in shots if s[1]-s[0]>=min_len]
    return shots, fi, fps


def extractFeatures(video_path, shots, h_bins=18, s_bins=8, v_bins=8):
    feat_dim = h_bins+s_bins+v_bins
    cap = cv2.VideoCapture(video_path)
    feats = []
    for start, end in shots:
        mid = (start+end) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            feats.append(np.zeros(feat_dim, np.float32))
            continue
        hsv = _bgr_to_hsv(frame)
        feat = np.concatenate([
            _calc_hist(hsv[..., 0], h_bins, (0, 180)),
            _calc_hist(hsv[..., 1], s_bins, (0, 256)),
            _calc_hist(hsv[..., 2], v_bins, (0, 256)),
        ])
        feat /= feat.sum()+1e-9
        feats.append(feat.astype(np.float32))
    cap.release()
    return np.array(feats)


def compute_dist_matrix(feats):
    """Compute symmetric pairwise L2 distance matrix between shot features."""
    N = len(feats)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.sqrt(np.sum((feats[i]-feats[j]) ** 2))
    return (dist+dist.T) / 2.0


def estimate_number_ofCuts(dist_mat):
    """Estimate number of scenes via SVD elbow on the distance matrix."""
    N =len(dist_mat)
    sv =np.linalg.svd(dist_mat, compute_uv=False)
    sv_log= np.log(sv[:max(2, int(N * 0.5))]+1e-9)
    c0 =sv_log[0]
    slope =(sv_log[-1]-sv_log[0]) / (len(sv_log)-1+1e-9)
    estimatedK =2
    max_d=0.0
    for i in range(len(sv_log)):
        d = abs(-sv_log[i] +c0 +slope*i)/math.sqrt(slope**2+1)
        if d > max_d:
            max_d = d
            estimatedK = i
    return max(2,min(estimatedK, N-1))


def normalizedCostClustering(N, K, dist_mat):
    ps = _prefix_sum(dist_mat)
    def dsum(a, b):
        return ps[b+1][b+1]-ps[a][b+1]-ps[b+1][a]+ps[a][a]

    cost = {}
    boundaries = {}
    area = {}

    k = 1
    for n in range(N):
        area = (N-n) ** 2
        
        for p in _area_sums(n, K-k):
            cost[(n, k, p)] =dsum(n, N-1)/(p+area)
            boundaries[(n, k, p)] =N-1

            area[(n, k, p)]= area

    for k in range(2, K+1):

        for n in range(N-1):
            if (N-n) < k:
                continue

            for p in _area_sums(n, K-k):
                best_cost=float('inf')
                best_i=-1

                for i in range(n, N-1):
                    if (N-i) < k:
                        continue
                    area = (i- n+1)**2
                    nextKey = (i+1, k- 1, p+area)
                    g = dsum(n, i)/(p +area+area.get(nextKey,0))
                    cost = g+cost.get(nextKey, 0)
                    if cost <best_cost:
                        best_cost= cost
                        best_i =i
                if best_i >= 0:
                    cost[(n, k, p)] =best_cost
                    boundaries[(n, k, p)]= best_i
                    area[(n, k, p)]=(best_i-n+1)**2+area.get((best_i+1, k-1, p +(best_i-n+1)**2), 0)
    return boundaries


def getCuts(J, K):
    n=0
    p_acc=0
    groups =[]
    for step in range(K):
        key =(n,K-step,p_acc)
        if key not in J:
            break
        last=J[key]
        groups.append((n, last))
        p_acc +=(last-n+1) ** 2
        n =last+1
    return groups


class SceneSegmenter:
    def __init__(self, video_path, threshold=30, min_shot_len=10):
        self.video_path =video_path
        self.threshold= threshold
        self.min_shot_len= min_shot_len
        self.scenes_list =[]
        self.middleFrames = []

    def segment_video(self, threshold=None):
        if threshold is None:
            threshold = self.threshold
        self.scenes_list =[]
        self.middleFrames= []
        shots,total_frames,fps=detectShots(self.video_path,threshold, self.min_shot_len)
        feats=extractFeatures(self.video_path, shots)
        dist_mat=compute_dist_matrix(feats)
        num_scenes=estimate_number_ofCuts(dist_mat)
        N=len(shots)
        boundaries = normalizedCostClustering(N,num_scenes,dist_mat)
        groups=getCuts(boundaries, num_scenes)

        intervals = [(shots[f][0], shots[l][1]) for f, l in groups]
        if intervals:
            intervals[-1] = (intervals[-1][0], total_frames-1)

        for i, (start_f, end_f) in enumerate(intervals):
            self.scenes_list.append(Scene(
                index=i,
                start_time=start_f / fps,
                end_time=end_f / fps,
                start_frame=start_f,
                end_frame=end_f,
                fps=fps,
            ))

        return self.scenes_list

    def calculate_frames_to_extract(self, scene):
        """
        Calculate which frames to extract based on scene duration.
        """
        duration = scene.duration
        start_frame = scene.start_frame
        end_frame = scene.end_frame
        total_frames = end_frame-start_frame
        if total_frames <= 0:
            return [start_frame]
        if duration < 2.0:
            return [start_frame+total_frames // 2]
        if duration < 5.0:
            return [
            start_frame+int(total_frames * 0.33),
                start_frame+int(total_frames*0.67),
            ]

        if duration < 10.0:
            return [
            start_frame+int(total_frames * 0.25),
                start_frame+int(total_frames * 0.50),
                start_frame+int(total_frames * 0.75),
            ]

        frames_per_interval = max(1, int(scene.fps * 3))

        num_frames = min(5, max(1, total_frames // frames_per_interval))
        
        return [
            start_frame+int(total_frames * (i+1) / (num_frames+1))
            for i in range(num_frames)
        ]

    def extract_frames(self):
        """
        Extract representative frames from each detected scene.

        Returns a list of dicts with keys: scene_index, frame_number, frame,
        scene, frame_count_in_scene.
        """
        if not self.scenes_list:
            return []

        self.middleFrames = []
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: could not open {self.video_path}")
            return []

        for scene in self.scenes_list:
            frame_numbers = self.calculate_frames_to_extract(scene)
            for frame_num in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    self.middleFrames.append({
                    'scene_index': scene.index,
                        'frame_number': frame_num,
                        'frame': frame,
                        'scene': scene,
                            'frame_count_in_scene': len(frame_numbers),
                    })
                else:
                    print(f"Warning: could not read frame {frame_num} (scene {scene.index})")

        cap.release()
        print(f"Extracted {len(self.middleFrames)} frames from {len(self.scenes_list)} scenes")
        return self.middleFrames

    def get_frames_for_ocr(self):
        if not self.middleFrames:
            self.extract_frames()
        return[d['frame'] for d in self.middleFrames]

    def get_frames_with_metadata(self):
        if not self.middleFrames:
            self.extract_frames()
        return self.middleFrames

    def get_scenes(self):
        return self.scenes_list
