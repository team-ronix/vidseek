import cv2
import json
import numpy as np
import os
import joblib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..utils.ovo_svm import OvO_SVM
from ..utils.Hog import HoG, calc_gradients, predict_char
from ..utils.MSER import merge_boxes, sort_word_chars, remove_image_border_box, remove_holes, remove_large_boxes
from ..utils.text_detector import extract_word_images
from ..utils.craft import CRAFT, extract_word_images_craft
from Storage.SQL.Repositories.OCRRepository import OCRRepository

_OCR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EAST_MODEL_PATH = os.path.join(_OCR_DIR, 'models', 'frozen_east_text_detection.pb')
_SVM_MODEL_DIR = os.path.join(_OCR_DIR, 'models', 'from_scratch_SVM')
_CRAFT_MODEL_PATH = os.path.join(_OCR_DIR, 'models', 'CRAFT_dynamic_5k', 'craft_epoch4_dynamic_5k.pth')


class OCR:
    def __init__(self, frames, video_path, video_id,
                 detector='craft', recognizer='easyocr'):
        """
        detector  : 'east' | 'craft'
        recognizer: 'mser' | 'easyocr'
        """
        self.frames = frames
        self.video_path = video_path
        self.inverted_index = {}
        self.video_id = video_id
        self.detector = detector
        self.recognizer = recognizer

        #  Text detector 
        if detector == 'east':
            self.net = cv2.dnn.readNet(_EAST_MODEL_PATH)

        elif detector == 'craft':
            import torch
            self.craft_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.craft_model = CRAFT()
            ckpt = torch.load(_CRAFT_MODEL_PATH, map_location=self.craft_device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.craft_model.load_state_dict(state_dict)
            self.craft_model.to(self.craft_device)
            self.craft_model.eval()
            print(f"CRAFT loaded on {self.craft_device}")

        else:
            raise ValueError(f"Unknown detector {detector!r}. Choose 'east' or 'craft'.")

        #  Text recognizer 
        if recognizer == 'mser':
            self.mser = cv2.MSER_create()
            self.model = OvO_SVM().load(_SVM_MODEL_DIR)
            self.le = joblib.load(os.path.join(_SVM_MODEL_DIR, 'OvO_SVM_label_encoder.joblib'))

        elif recognizer == 'easyocr':
            import torch
            import easyocr
            gpu = torch.cuda.is_available()
            self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu)
            print(f"EasyOCR loaded (gpu={gpu})")

        else:
            raise ValueError(f"Unknown recognizer {recognizer!r}. Choose 'mser' or 'easyocr'.")

        self._ocr_repo = OCRRepository()

    #  Detection helpers 

    def _detect_words(self, frame):
        if self.detector == 'east':
            return extract_word_images(frame, self.net, pad=8)
        else:
            return extract_word_images_craft(frame, self.craft_model, self.craft_device)

    #  Recognition helpers 

    def _recognize_word_mser(self, word_img):
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary == 0) > np.sum(binary == 255):
            binary = cv2.bitwise_not(binary)

        _, boxes = self.mser.detectRegions(binary)
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

        word_text = text.strip().lower()
        word_confidence = float(np.mean(char_confidences)) if char_confidences else 0.0
        return word_text, word_confidence

    def _recognize_word_easyocr(self, word_img):
        results = self.easyocr_reader.readtext(word_img, detail=1)
        if not results:
            return '', 0.0
        text = ' '.join(r[1] for r in results).strip().lower()
        confidence = float(np.mean([r[2] for r in results]))
        if confidence < 0.5:
            return None, 0.0
        return text, confidence

    def _recognize_word(self, word_img):
        if self.recognizer == 'mser':
            return self._recognize_word_mser(word_img)
        else:
            return self._recognize_word_easyocr(word_img)

    #  Main loop 

    def process_frames(self):
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']

            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"Processing frame {frame_number}")

            if frame_number is None:
                continue

            if frame is not None:
                print(f"Running OCR (detector={self.detector}, recognizer={self.recognizer})...")
                word_images = self._detect_words(frame)
                print(f"Detected {len(word_images)} word regions")

                for word_img in word_images:
                    text, confidence = self._recognize_word(word_img)

                    if not text or len(text.strip()) < 2:
                        continue

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

                    self._ocr_repo.save_word(
                        word=text,
                        video_id=self.video_id,
                        start_time=scene.start_time,
                        end_time=scene.end_time,
                        frame_number=frame_number,
                        word_detection_model = self.detector,
                        word_recognition_model = self.recognizer
                    )

            else:
                print("Warning: Frame data is None")

    def save_inverted_index(self, output_path='ocr_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)

    def get_inverted_index(self):
        return self.inverted_index
