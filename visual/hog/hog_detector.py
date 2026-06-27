from pathlib import Path
import shutil
import gc
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from visual.hog.features.hog_descriptor import HOGDescriptor
from visual.hog.datastructures.voc_dataset import VOCDataset
from visual.hog.datastructures.component import Component
from visual.hog.utils import calculate_iou
from visual.hog.persistence import save, load
from visual.hog.checkpoint import save_checkpoint, load_checkpoint
from visual.hog.training.data_processing import process_dataset, learn_window_sizes
from visual.hog.training.feature_construction import construct_Xpos_Xneg
from visual.hog.training.latent_update import update_positive_latents, fit_bbox_regs
from visual.hog.training.hard_negative_mining import mine_hard_negatives_pyramid
from visual.hog.inference.detection import detect
from visual.hog.inference.contextual_rescoring import ContextualRescorer

class HOGDetector:
    def __init__(
        self,
        classes,
        hog_descriptor_params,
        n_components=2,
        c_svm=0.01,
        max_itr_svm=30000,
        training_epochs=2,
        area_percentile=80,
        hard_neg_threshold=-1,
        pyramid_step=2,
        max_hard_per_image=20,
        bg_multiplier=2.0,
        other_classes_total_ratio=1.0,
        bbr_alpha=1000.0,
        min_iou_between_gt_and_latent=0.35,
        contextual_rescorer_iou_thresh=0.5,
        contextual_rescorer_C=1.0,
        contextual_rescorer_detection_threshold=0.05,
        contextual_rescorer_neg_size=30000,
    ):
        self.classes = classes
        self.hog_params = hog_descriptor_params
        self.n_components = n_components
        self.c_svm = c_svm
        self.max_itr_svm = max_itr_svm
        self.training_epochs = training_epochs
        self.area_percentile = area_percentile
        self.hard_neg_threshold = hard_neg_threshold
        self.pyramid_step = pyramid_step
        self.max_hard_per_image = max_hard_per_image
        self.bg_multiplier = bg_multiplier
        self.other_classes_total_ratio = other_classes_total_ratio
        self.bbr_alpha = bbr_alpha
        self.min_iou_between_gt_and_latent = min_iou_between_gt_and_latent
        self.default_window_size = (64, 64)
        self.hog_descriptor = HOGDescriptor(**self.hog_params)
        self.cls_comps = {cls: [] for cls in self.classes}
        self.svms = {cls: [] for cls in self.classes}
        self.kmeans_clfs = {cls: None for cls in self.classes}
        self.trained_flag = False
        self.contextual_rescorer = ContextualRescorer(
            classes=self.classes,
            iou_thresh=contextual_rescorer_iou_thresh,
            C=contextual_rescorer_C,
            detection_threshold=contextual_rescorer_detection_threshold,
            neg_size=contextual_rescorer_neg_size,
        )

    def _extract_custom_hog(self, image_patch):
        return self.hog_descriptor.compute_feature_map(image_patch).flatten()

    def _calculate_iou(self, boxA, boxB):
        return calculate_iou(boxA, boxB)

    def train(
        self,
        train_ds,
        min_box_area=400,
        max_train_images=None,
        max_rescore_images=None,
        neg_patches_per_image=2,
        checkpoint_path='./checkpoint',
        skip_step1=False,
        rescorer_checkpoint_every=50,
    ):
        checkpoint_path = Path(checkpoint_path)
        skip_step1 = skip_step1 and checkpoint_path.exists()
        if not skip_step1:
            t1 = time.time()
            print("\nStep 1: Initialisation\n")
            print("1-a: Processing dataset")
            pos_patches, bboxes_wh, gt_boxes, neg_images = process_dataset(self, train_ds, min_box_area, max_train_images)
            t2 = time.time()
            print(f"\tDone in {t2 - t1:.2f} seconds")
            print("\n1-b: Learning window sizes")
            comp_labels = learn_window_sizes(self, bboxes_wh)
            t3 = time.time()
            print(f"\tDone in {t3 - t2:.2f} seconds")
            print("\n1-c: Constructing X_pos and X_neg per component")
            print("\t(background patches sampled at each component's pixel size)\n")
            construct_Xpos_Xneg(self, pos_patches, comp_labels, neg_images, neg_patches_per_image)
            t4 = time.time()
            print(f"\tDone in {t4 - t3:.2f} seconds")
            print("\n1-d: Fitting initial SVMs")
            for comps in self.cls_comps.values():
                for comp in comps:
                    comp.fit_svm(split_ratio=None)
                    gc.collect()
            t5 = time.time()
            print(f"\tDone in {t5 - t4:.2f} seconds")
            save_checkpoint(self, checkpoint_path, pos_patches, cur_epoch=0)
        if skip_step1:
            epoch, pos_patches = load_checkpoint(self, checkpoint_path)
        else:
            epoch = 0
        t5 = time.time()
        print("\nStep 2: Latent SVM loop\n")
        while epoch < self.training_epochs:
            print(f"Epoch {epoch + 1}/{self.training_epochs}\n")
            print("2-a: Updating positive latent assignments")
            update_positive_latents(self, train_ds, max_train_images)
            t6 = time.time()
            print(f"\tDone in {t6 - t5:.2f} seconds")
            print("\n2-b: Mining hard negatives via feature pyramid")
            mine_hard_negatives_pyramid(self, train_ds, max_train_images, pos_patches, do_flush_initial=(epoch == 0))
            t7 = time.time()
            print(f"\tDone in {t7 - t6:.2f} seconds")
            print("\n2-c: Retraining component SVMs")
            split_ratio = 0.15 if epoch == self.training_epochs - 1 else None
            for cls, comps in self.cls_comps.items():
                for comp in comps:
                    comp.fit_svm(split_ratio=split_ratio)
                    gc.collect()
                self.svms[cls] = [comp.svm for comp in comps]
            t8 = time.time()
            print(f"\tDone in {t8 - t7:.2f} seconds")
            epoch += 1
            save_checkpoint(self, checkpoint_path, pos_patches, cur_epoch=epoch)
        t8 = time.time()
        print("\nStep 3: Post-processing\n")
        print("3-a: Bounding-box regression")
        fit_bbox_regs(self)
        t9 = time.time()
        print(f"\tDone in {t9 - t8:.2f} seconds")
        print("\n3-b: Score calibration")
        for comps in self.cls_comps.values():
            for comp in comps:
                comp.fit_calibration()
        t10 = time.time()
        print(f"\tDone in {t10 - t9:.2f} seconds")
        trained = [c for c in self.classes if self.svms.get(c)]
        for cls, comps in self.cls_comps.items():
            for comp in comps:
                comp.del_training_data()
        gc.collect()
        self.trained_flag = True
        t10 = time.time()
        print("\n3-c: Contextual Rescoring")
        rescorer_checkpoint_path = checkpoint_path / "contextual_rescorer_ckpt.pkl"
        self.contextual_rescorer.fit(
            self,
            train_ds,
            max_rescore_images if max_rescore_images != None else max_train_images,
            checkpoint_path=str(rescorer_checkpoint_path),
            checkpoint_every=rescorer_checkpoint_every,
        )
        t11 = time.time()
        print(f"\tDone in {t11 - t10:.2f} seconds")
        gc.collect()
        print(f"\nTraining complete - {len(trained)}/{len(self.classes)} classes, {self.n_components} components each.")
        # shutil.rmtree(checkpoint_path)

    def detect(self, image, threshold=None, overlap_threshold=0.3, pyramid_lambda=None, use_context=True):
        boxes, scores, labels = detect(self, image, overlap_threshold, pyramid_lambda)
        if use_context == True and self.contextual_rescorer.fitted == True and boxes:
            ih, iw = image.shape[:2]
            scores = self.contextual_rescorer.rescore(boxes, scores, labels, iw, ih)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            boxes = [boxes[i] for i in order]
            scores = [scores[i] for i in order]
            labels = [labels[i] for i in order]
        if threshold is None:
            threshold = 0
        keep_indices = np.where(np.array(scores) >= threshold)[0]
        boxes = [boxes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
        return boxes, scores, labels

    def save(self, path):
        save(self, path)

    def load(self, path):
        load(self, path)
