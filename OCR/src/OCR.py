import cv2
import json
import numpy as np
import os
import joblib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ovo_svm import OvO_SVM
from ..utils.Hog import HoG, calc_gradients, predict_char
from ..utils.MSER import merge_boxes, sort_word_chars, remove_image_border_box, remove_holes, remove_large_boxes
from ..utils.text_detector import extract_word_images

_OCR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EAST_MODEL_PATH = os.path.join(_OCR_DIR, 'models', 'frozen_east_text_detection.pb')
_SVM_MODEL_DIR = os.path.join(_OCR_DIR, 'models', 'from_scratch_SVM')

class OCR:
    def __init__(self, frames, video_path):
        self.frames = frames
        self.video_path = video_path
        self.inverted_index = {}

        self.net = cv2.dnn.readNet(_EAST_MODEL_PATH)
        self.mser = cv2.MSER_create()
        print(f"Loading SVM model from {_SVM_MODEL_DIR}...")
        self.model = OvO_SVM().load(_SVM_MODEL_DIR)
        self.le = joblib.load(os.path.join(_SVM_MODEL_DIR, 'OvO_SVM_label_encoder.joblib'))

    def _recognize_word(self, word_img):
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, boxes = self.mser.detectRegions(gray)

        unique_boxes = list({tuple(b) for b in boxes})
        unique_boxes = merge_boxes(unique_boxes, threshold=0.3)
        unique_boxes = sort_word_chars(unique_boxes)
        unique_boxes = remove_image_border_box(unique_boxes, word_img.shape)
        unique_boxes = remove_holes(unique_boxes)
        unique_boxes = remove_large_boxes(unique_boxes, word_img.shape)

        text = ""
        char_confidences = []
        for box in unique_boxes:
            x, y, w, h = box
            char_img = gray[y:y+h, x:x+w]
            if char_img.size == 0:
                continue

            padded = cv2.copyMakeBorder(char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            blurred = cv2.GaussianBlur(padded, (5, 5), 0)
            binarized = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            resized = cv2.resize(binarized, (128, 128), interpolation=cv2.INTER_CUBIC)
            dilated = cv2.dilate(resized, np.ones((3, 3), np.uint8), iterations=1)
            normalized = dilated.astype(np.float32) / 255.0

            magnitudes, orientations = calc_gradients(normalized)
            features = HoG(orientations, magnitudes)
            predicted_label, confidence = predict_char(features, self.model, self.le)
            text += self.le.inverse_transform(predicted_label)[0]
            if type(confidence) == np.ndarray or type(confidence) == list:
                confidence = confidence[0]
            char_confidences.append(float(confidence))

        word_text = text.strip().lower()
        word_confidence = float(np.mean(char_confidences)) if char_confidences else 0.0
        return word_text, word_confidence

    def process_frames(self):
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']

            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"       Processing frame {frame_number}")

            if frame_number is None:
                continue

            if frame is not None:
                print("       Running OCR...")
                word_images = extract_word_images(frame, self.net)
                print(f"       Detected {len(word_images)} word regions")

                for word_img in word_images:
                    text, confidence = self._recognize_word(word_img)

                    if not text or confidence < 0.5 or len(text.strip()) < 5:
                        continue

                    # print(f"       - '{text}' (confidence: {confidence:.2f})")

                    if text not in self.inverted_index:
                        self.inverted_index[text] = []

                    if any(occ['scene'] == scene.index for occ in self.inverted_index[text]):
                        continue

                    self.inverted_index[text].append({
                        'scene': scene.index,
                        'frame': frame_number,
                        'video_path': self.video_path,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time,
                        'confidence': confidence
                    })
            else:
                print("       Warning: Frame data is None")

    def save_inverted_index(self, output_path='ocr_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)

    def get_inverted_index(self):
        return self.inverted_index
