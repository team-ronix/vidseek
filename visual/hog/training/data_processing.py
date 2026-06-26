import gc
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from visual.hog.datastructures.voc_dataset import VOCDataset
from visual.hog.datastructures.component import Component


def _min_component_positives(n_samples: int) -> int:
    return max(15, min(40, int(0.10 * n_samples)))


def process_dataset(
    detector,
    dataset: VOCDataset,
    min_box_area: int = 400,
    max_images: int | None = None,
) -> tuple[dict, dict, dict, list]:
    pos_patches: dict[str, list[np.ndarray]] = {cls: [] for cls in detector.classes}
    bboxes_wh: dict[str, list[tuple[int, int]]] = {cls: [] for cls in detector.classes}
    gt_boxes: dict[str, list[tuple[float, float, float, float]]] = {cls: [] for cls in detector.classes}
    neg_images: list[tuple[str, list]] = []
    n = len(dataset) if max_images is None else min(len(dataset), max_images)
    indices = np.random.permutation(n)
    for loop_index, index in enumerate(tqdm(indices, desc="Extracting patches")):
        img = dataset.get_image(index)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        boxes, lbls = dataset.get_annotation(index)
        for (xmin, ymin, xmax, ymax), cls in zip(boxes, lbls):
            if cls not in detector.classes:
                continue
            x0, y0 = max(0, int(xmin)), max(0, int(ymin))
            x1, y1 = min(iw, int(xmax)), min(ih, int(ymax))
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            if w * h < min_box_area:
                continue
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            pos_patches[cls].append(patch.copy())
            bboxes_wh[cls].append((w, h))
            gt_boxes[cls].append((float(x0), float(y0), float(x1), float(y1)))
            del patch
        all_gt = [
            (float(max(0, int(b[0]))), float(max(0, int(b[1]))),
             float(min(iw, int(b[2]))), float(min(ih, int(b[3]))))
            for b in boxes
        ]
        if ih >= 32 and iw >= 32:
            img_path = dataset._img_path(dataset.image_ids[index])
            if img_path is not None:
                neg_images.append((img_path, all_gt))
        del img, boxes, lbls
        if (loop_index + 1) % 100 == 0:
            gc.collect()
    print("\nPositive patches per class:")
    for cls in detector.classes:
        print(f"{cls}: {len(pos_patches[cls]):5d}  (raw bboxes: {len(bboxes_wh[cls])})")
    print(f"\nBackground image paths collected: {len(neg_images)}")
    gc.collect()
    return dict(pos_patches), dict(bboxes_wh), dict(gt_boxes), neg_images


def learn_window_sizes(detector, bboxes_wh: dict) -> dict:
    comp_labels: dict[str, np.ndarray] = {cls: np.zeros(0, dtype=int) for cls in detector.classes}
    for cls in detector.classes:
        whs_cls = np.array(bboxes_wh[cls]) if bboxes_wh[cls] else np.empty((0, 2))
        if len(whs_cls) == 0:
            print(
                f"Warning: No bounding boxes for class '{cls}', "
                f"using default window size {detector.default_window_size}."
            )
            detector.cls_comps[cls].append(
                _make_component(detector, comp_id=0, cls=cls)
            )
            comp_labels[cls] = np.zeros(0, dtype=int)
            continue
        aspect_ratios = whs_cls[:, 0] / whs_cls[:, 1]
        n_samples = len(aspect_ratios)
        min_pos = _min_component_positives(n_samples)
        k = min(detector.n_components, n_samples)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw_ids = kmeans.fit_predict(aspect_ratios.reshape(-1, 1))
        detector.kmeans_clfs[cls] = kmeans
        unique_labels, counts = np.unique(raw_ids, return_counts=True)
        label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
        surviving = [
            lbl for lbl in unique_labels
            if label_counts[lbl] >= min_pos
        ]
        sparse = [
            lbl for lbl in unique_labels
            if label_counts[lbl] < min_pos
        ]
        if sparse:
            print(
                f"[{cls}] Merging {len(sparse)} sparse cluster(s) "
                f"(< {min_pos} samples, adaptive threshold for {n_samples} total) "
                f"into nearest surviving cluster."
            )
        if not surviving:
            surviving = [unique_labels[np.argmax(counts)]]
            sparse = [lbl for lbl in unique_labels if lbl not in surviving]
            print(f"[{cls}] All clusters sparse - keeping largest cluster only.")
        surviving_centroids = np.array([kmeans.cluster_centers_[lbl][0] for lbl in surviving])
        final_ids = raw_ids.copy()
        for sp_lbl in sparse:
            sp_centroid = kmeans.cluster_centers_[sp_lbl][0]
            nearest_surviving = surviving[int(np.argmin(np.abs(surviving_centroids - sp_centroid)))]
            final_ids[raw_ids == sp_lbl] = nearest_surviving
        for orig_label in surviving:
            members = aspect_ratios[final_ids == orig_label]
            if len(members) > 0:
                kmeans.cluster_centers_[orig_label][0] = float(members.mean())
                print(
                    f"[{cls}] Cluster {orig_label}: centroid updated to "
                    f"{kmeans.cluster_centers_[orig_label][0]} "
                    f"(from {len(members)} merged members)"
                )
        present_labels = sorted(set(final_ids.tolist()))
        label_to_comp_id = {lbl: cid for cid, lbl in enumerate(present_labels)}
        remapped_ids = np.array([label_to_comp_id[l] for l in final_ids], dtype=int)
        comp_labels[cls] = remapped_ids
        for comp_id, orig_label in enumerate(present_labels):
            comp_bboxes_whs = whs_cls[final_ids == orig_label]
            member_count = len(comp_bboxes_whs)
            if member_count == 0:
                print(
                    f"Warning: Empty cluster for class '{cls}' component {comp_id}, "
                    f"using default window size {detector.default_window_size}."
                )
                detector.cls_comps[cls].append(
                    _make_component(detector, comp_id=comp_id, cls=cls)
                )
                continue
            comp_aspect = np.mean(comp_bboxes_whs[:, 0] / comp_bboxes_whs[:, 1])
            areas = comp_bboxes_whs[:, 0] * comp_bboxes_whs[:, 1]
            comp_area = np.percentile(areas, detector.area_percentile)
            comp_h = np.round(np.sqrt(comp_area / comp_aspect))
            comp_w = np.round(comp_h * comp_aspect)
            cell = detector.hog_descriptor.cell_size
            detector.cls_comps[cls].append(Component(
                component_id=comp_id,
                class_name=cls,
                cell_w=max(1, int(comp_w // cell)),
                cell_h=max(1, int(comp_h // cell)),
                cell_size=cell,
                c_svm=detector.c_svm,
                max_itr_svm=detector.max_itr_svm,
                alpha=detector.bbr_alpha,
            ))
            print(
                f"Learned window for class '{cls}' comp {comp_id}: "
                f"({comp_w:.0f}, {comp_h:.0f}) px [{member_count} positives]"
            )
    return comp_labels


def _make_component(detector, comp_id: int, cls: str) -> Component:
    cell = detector.hog_descriptor.cell_size
    return Component(
        component_id=comp_id,
        class_name=cls,
        cell_w=max(1, detector.default_window_size[0] // cell),
        cell_h=max(1, detector.default_window_size[1] // cell),
        cell_size=cell,
        c_svm=detector.c_svm,
        max_itr_svm=detector.max_itr_svm,
        alpha=detector.bbr_alpha,
    )