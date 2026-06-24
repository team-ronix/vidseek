import cv2
import numpy as np
from typing import Tuple
from visual.hog.datastructures.feature_level import FeatureLevel
from concurrent.futures import ThreadPoolExecutor

# n_orient_cs: orientation bins for contrast-sensitive features (0-360 degrees)
# n_orient_ci: orientation bins for contrast-insensitive features (0-180 degrees)
# alpha: truncation threshold for normalized cell features
class HOGDescriptor:
    def __init__(
            self, cell_size: int = 8, 
            n_orient_cs: int = 18, n_orient_ci: int = 9,
            alpha: float = 0.2, n_energy: int = 4,
            n_octaves=5, llambda=5, min_size=32
        ):
        self.cell_size = cell_size
        self.n_orient_cs = n_orient_cs
        self.n_orient_ci = n_orient_ci
        self.alpha = alpha
        self.n_energy = n_energy
        self.n_octaves = n_octaves
        self.llambda = llambda
        self.min_size = min_size
        self.feature_dim = n_orient_cs + n_orient_ci + n_energy
        
    def _process_img(self, img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        processed_img = img.astype(np.float32)
        if img.dtype == np.uint8:
            processed_img /= 255.0
        return processed_img

    def _compute_gradients(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W, C = img.shape
        dx, dy = np.zeros_like(img), np.zeros_like(img)
        # non-border pixels
        dx[:, 1:-1, :] = img[:, 2:, :] - img[:, :-2, :]
        dy[1:-1, :, :] = img[2:, :, :] - img[:-2, :, :]
        # borders (replicate the one-sided difference)
        dx[:, 0, :] = img[:, 1, :] - img[:, 0, :]
        dx[:, -1, :] = img[:, -1, :] - img[:, -2, :]
        dy[0, :, :] = img[1, :, :] - img[0, :, :]
        dy[-1, :, :] = img[-1, :, :] - img[-2, :, :]
        channels_mag = np.sqrt(dx ** 2 + dy ** 2)
        # for each pixel, find the channel with the maximum gradient magnitude
        best_channel = np.argmax(channels_mag, axis=2) 
        rows, cols = np.mgrid[:H, :W]
        mag = channels_mag[rows, cols, best_channel]
        gx = dx[rows, cols, best_channel]
        gy = dy[rows, cols, best_channel]
        orient = np.mod(np.arctan2(gy, gx), 2 * np.pi)
        return mag, orient

    def _pixel_feature_map(self, mag, orient) -> Tuple[np.ndarray, np.ndarray]:
        mgH, mgW = mag.shape
        rows, cols = np.mgrid[:mgH, :mgW]
        cs_bins = np.mod(np.round(orient / (2 * np.pi) * self.n_orient_cs).astype(int), self.n_orient_cs)
        ci_bins = np.mod(np.round(orient / np.pi * self.n_orient_ci).astype(int), self.n_orient_ci)
        F_cs = np.zeros((mgH, mgW, self.n_orient_cs), dtype=np.float32)
        F_ci = np.zeros((mgH, mgW, self.n_orient_ci), dtype=np.float32)
        F_cs[rows, cols, cs_bins] = mag
        F_ci[rows, cols, ci_bins] = mag
        return F_cs, F_ci
    
    def _spatial_aggregate(self, F: np.ndarray) -> np.ndarray:
        fH, fW, fD = F.shape
        cH = (fH - 1) // self.cell_size + 1
        cW = (fW - 1) // self.cell_size + 1

        rows, cols = np.mgrid[:fH, :fW]
        cx = (cols / self.cell_size - 0.5).clip(0)
        cy = (rows / self.cell_size - 0.5).clip(0)

        ix0 = cx.astype(np.int32).clip(0, cW - 1)
        ix1 = (ix0 + 1).clip(0, cW - 1)
        wx1 = (cx - ix0).astype(np.float32).clip(0, 1)
        wx0 = 1.0 - wx1

        iy0 = cy.astype(np.int32).clip(0, cH - 1)
        iy1 = (iy0 + 1).clip(0, cH - 1)
        wy1 = (cy - iy0).astype(np.float32).clip(0, 1)
        wy0 = 1.0 - wy1

        C = np.zeros((cH * cW, fD), dtype=np.float32)
        # Reuse a single (fH*fW, fD) buffer for all four bilinear corners
        # instead of allocating a fresh full copy per corner.
        buf = np.empty((fH * fW, fD), dtype=np.float32)
        four_corners = [
            (iy0 * cW + ix0, wx0 * wy0),
            (iy0 * cW + ix1, wx1 * wy0),
            (iy1 * cW + ix0, wx0 * wy1),
            (iy1 * cW + ix1, wx1 * wy1),
        ]
        F_flat = F.reshape(-1, fD)
        for flat_idx, w in four_corners:
            # Write weighted pixels into buf in-place - no extra allocation.
            np.multiply(F_flat, w.ravel()[:, np.newaxis], out=buf)
            np.add.at(C, flat_idx.ravel(), buf)
        return C.reshape(cH, cW, fD)

    def _normalize(self, C: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        cH, cW, _ = C.shape
        energy = (C ** 2).sum(axis=2)
        dx_dy_pairs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        N = np.zeros((cH, cW, self.n_energy), dtype=np.float32)
        rows = np.arange(cH)
        cols = np.arange(cW)
        for k, (dx, dy) in enumerate(dx_dy_pairs):
            shift_r = np.clip(rows + dx, 0, cH - 1)
            shift_c = np.clip(cols + dy, 0, cW - 1)
            energy_shifted_r = energy[shift_r, :]
            energy_shifted_c = energy[:, shift_c]
            energy_shifted_rc = energy[(shift_r.reshape(-1, 1), shift_c)]
            N[:, :, k] = np.sqrt(energy + energy_shifted_r + energy_shifted_c + energy_shifted_rc + epsilon)
        return N
    
    def _feature_analysis(self, C_CI: np.ndarray, C_CS: np.ndarray, N: np.ndarray) -> np.ndarray:
        nH, nW, _ = C_CI.shape
        G = np.zeros((nH, nW, self.feature_dim), dtype=np.float32)
        for i in range(self.n_energy):
            n = N[:, :, i]
            n = n[:, :, np.newaxis]
            ci_norm = np.clip(C_CI / n, 0, self.alpha)
            cs_norm = np.clip(C_CS / n, 0, self.alpha)
            G[:, :, :self.n_orient_ci] += ci_norm
            G[:, :, self.n_orient_ci:self.n_orient_ci + self.n_orient_cs] += cs_norm
            G[:, :, self.n_orient_ci + self.n_orient_cs + i] = ci_norm.sum(axis=2)
        return G

    def compute_feature_map(self, image:np.ndarray) -> np.ndarray:
        img = self._process_img(image)
        mag, orient = self._compute_gradients(img)
        F_cs, F_ci = self._pixel_feature_map(mag, orient)
        C_cs = self._spatial_aggregate(F_cs)
        C_ci = self._spatial_aggregate(F_ci)
        N = self._normalize(C_ci)
        G = self._feature_analysis(C_ci, C_cs, N)
        return G

    def compute_feature_pyramid(self, image:np.ndarray, llambda=None) -> list:
        cur_lambda = llambda if llambda is not None else self.llambda
        img = self._process_img(image)
        H, W = img.shape[:2]
 
        level_specs = []
        # Generate up-scaled and downscaled versions of the image
        for l in range(-cur_lambda, self.n_octaves * cur_lambda):
            scale = 2 ** (l / cur_lambda)
            new_H = int(round(H / scale))
            new_W = int(round(W / scale))
            if new_H < self.min_size or new_W < self.min_size:
                break
            if new_H > 2 * H or new_W > 2 * W:
                continue
            level_specs.append((l, scale, new_H, new_W, img))
 
        max_workers = min(len(level_specs), (cv2.getNumberOfCPUs() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pyramid = list(executor.map(self._process, level_specs))
 
        return pyramid
    
    def _process(self, spec):
        l, scale, new_H, new_W, img = spec
        resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        feat = self.compute_feature_map(resized)
        return FeatureLevel(feature_map=feat, scale=scale, level=l)