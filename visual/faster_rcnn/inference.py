import argparse
import torch
from visual.faster_rcnn.model.faster_rcnn import FasterRCNN
from visual.faster_rcnn.voc_dataset import VOC_CLASSES
from PIL import Image
import numpy as np
import os


def process_img(img, target_size=600, max_size=1000):
    pixel_mean = np.array([0.485, 0.456, 0.406], np.float32)
    pixel_std = np.array([0.229, 0.224, 0.225], np.float32)
    w, h = img.size
    scale = target_size / min(h, w)
    if scale * max(h, w) > max_size:
        scale = max_size / max(h, w)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    arr = (np.array(img, np.float32) / 255.0 - pixel_mean) / pixel_std
    return torch.from_numpy(arr.copy()).permute(2, 0, 1), scale

def infer(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {DEVICE}')
    NUM_CLASSES = len(VOC_CLASSES)
    if args.image_path is None or not os.path.exists(args.image_path):
        raise ValueError("Image path is not specified or file does not exist")
    img = Image.open(args.image_path).convert("RGB")
    img, scale = process_img(img)
    model = FasterRCNN(num_classes=NUM_CLASSES, score_thresh=args.score_thresh).to(DEVICE)
    if args.model_path is None or not os.path.exists(args.model_path):
        raise ValueError("Model path is not specified or file does not exist")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    scaled_H, scaled_W = img.shape[1], img.shape[2]
    with torch.no_grad():
        res = model(img.unsqueeze(0).to(DEVICE), [(scaled_H, scaled_W)])[0]
    pred_boxes  = res['boxes'].cpu()
    pred_scores = res['scores'].cpu()
    pred_labels = res['labels'].cpu()
    print(f'\nDetections ({len(pred_scores)} found):')
    results = []
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        cls_name = VOC_CLASSES[label.item() - 1]
        x1, y1, x2, y2 = (v / scale for v in box.tolist())
        print(f'  {cls_name:<16s} {score:.3f}  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]')
        results.append((cls_name, score, (x1, y1, x2, y2)))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", help="Path to the input image")
    parser.add_argument("--model-path", help="Path to the trained model")
    parser.add_argument("--score-thresh", help="Score threshold for detections", type=float, default=0.5)
    args = parser.parse_args()
    infer(args)