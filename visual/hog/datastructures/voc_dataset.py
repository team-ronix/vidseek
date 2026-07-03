import os
import xml.etree.ElementTree as ET
import cv2



class VOCDataset:
    def __init__(self, root, image_set_file=None, class_to_idx=None):
        self.root = root
        self.img_dir = self._find_sub(root, ['JPEGImages', 'images', 'Images'])
        self.annot_dir = self._find_sub(root, ['Annotations', 'annotations'])
        if class_to_idx != None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {}
        if self.img_dir is None or self.annot_dir is None:
            raise FileNotFoundError(f'Missing JPEGImages or Annotations in {root}')
        if image_set_file and os.path.exists(image_set_file):
            with open(image_set_file) as f:
                self.image_ids = [l.strip().split()[0] for l in f if l.strip()]
            print(f'Loaded {len(self.image_ids)} IDs from split file.')
        else:
            self.image_ids = [os.path.splitext(fn)[0] for fn in os.listdir(self.annot_dir) if fn.endswith('.xml')]
            print(f'Directory fallback: {len(self.image_ids)} XML annotations found.')
        valid_ids = []
        for image_id in self.image_ids:
            img_path = self._img_path(image_id)
            ann_path = os.path.join(self.annot_dir, f'{image_id}.xml')
            if img_path != None and os.path.exists(ann_path):
                valid_ids.append(image_id)
        self.image_ids = valid_ids
        print(f'Final valid samples: {len(self.image_ids)}')

    def __len__(self):
        return len(self.image_ids)

    def get_image(self, idx):
        path = self._img_path(self.image_ids[idx])
        img = cv2.imread(path)
        return img

    def get_annotation(self, idx):
        image_id = self.image_ids[idx]
        annot_path = os.path.join(self.annot_dir, f'{image_id}.xml')
        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes = []
        lbls = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name not in self.class_to_idx:
                continue
            bb = obj.find('bndbox')
            xmin = float(bb.find('xmin').text)
            ymin = float(bb.find('ymin').text)
            xmax = float(bb.find('xmax').text)
            ymax = float(bb.find('ymax').text)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                lbls.append(name)
        return boxes, lbls

    def _find_sub(self, root, candidates):
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                return p
        return None

    def _img_path(self, img_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(self.img_dir, f'{img_id}{ext}')
            if os.path.exists(p):
                return p
        return None
