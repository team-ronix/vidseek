import os, random, xml.etree.ElementTree as ET
import torch, numpy as np
from torch.utils.data import Dataset
from PIL import Image

VOC_CLASSES = (
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
)
CLASS_TO_IDX = {c: i+1 for i, c in enumerate(VOC_CLASSES)}

class VOCDataset(Dataset):
    def __init__(
        self,
        base,
        split="trainval",
        target_size=600,
        max_size=1000,
    ):
        self.split = split
        self.flip_prob = 0.5 if split in {"train", "trainval"} else 0.0
        self.img_dir = os.path.join(base, "JPEGImages")
        self.ann_dir = os.path.join(base, "Annotations")
        split_file = os.path.join(base, "ImageSets", "Main", f"{split}.txt")
        with open(split_file) as f:
            self.ids = [l.split()[0] for l in f if l.strip()]

        self.target_size = target_size
        self.max_size = max_size
        self.pixel_mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.pixel_std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_t, scale = self._load_img(img_id)
        boxes, labels, difficults = self._load_ann(img_id, scale)
        if self.flip_prob and random.random() < self.flip_prob:
            img_t = torch.flip(img_t, dims=[2])
            if boxes.numel() > 0:
                width = img_t.shape[2]
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
        return img_t, boxes, labels, difficults, img_id

    def _load_img(self, img_id):
        img = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        w, h = img.size
        scale = self.target_size / min(h, w)
        if scale * max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        arr = (np.array(img, np.float32) / 255.0 - self.pixel_mean) / self.pixel_std
        return torch.from_numpy(arr.copy()).permute(2, 0, 1), scale

    def _load_ann(self, img_id, scale):
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        if not os.path.exists(ann_path):
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.bool)
            )
        boxes, labels, difficults = [], [], []
        for obj in ET.parse(ann_path).getroot().findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue
            diff = obj.find("difficult")
            is_diff = int(diff.text) if diff is not None else 0
            bb = obj.find("bndbox")
            x1 = (float(bb.find("xmin").text) - 1) * scale
            y1 = (float(bb.find("ymin").text) - 1) * scale
            x2 = (float(bb.find("xmax").text) - 1) * scale
            y2 = (float(bb.find("ymax").text) - 1) * scale
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[name])
            difficults.append(is_diff)
        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(difficults, dtype=torch.bool)
            )
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool)
        )

def collate_fn(batch):
    imgs, boxes_l, labels_l, difficults_l, ids = zip(*batch)
    max_h = max(i.shape[1] for i in imgs)
    max_w = max(i.shape[2] for i in imgs)
    padded, shapes = [], []
    for img in imgs:
        c, h, w = img.shape
        pad = torch.zeros(c, max_h, max_w, dtype=img.dtype)
        pad[:, :h, :w] = img
        padded.append(pad)
        shapes.append((h, w))
    return torch.stack(padded), shapes, list(boxes_l), list(labels_l), list(difficults_l), list(ids)
