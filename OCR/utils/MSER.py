def merge_boxes(boxes, threshold=0.3):
    merged = []

    for box in boxes:
        x, y, w, h = box
        if w == 0 or h == 0:
            continue

        added = False

        for i, (mx, my, mw, mh) in enumerate(merged):

            xi1 = max(x, mx)
            yi1 = max(y, my)
            xi2 = min(x + w, mx + mw)
            yi2 = min(y + h, my + mh)

            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            area1 = w * h
            area2 = mw * mh
            union = area1 + area2 - inter

            iou = inter / (union + 1e-6)

            if iou > threshold:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny

                merged[i] = (nx, ny, nw, nh)
                added = True
                break

        if not added:
            merged.append((x, y, w, h))

    return merged



def filter_char_boxes(boxes, lower_bound=0.1, higher_bound=2.5):
    filtered = []
    for box in boxes:
        x, y, w, h = box
        if w == 0 or h == 0:
            continue
        aspect_ratio = w / (h + 1e-6)
        if lower_bound <= aspect_ratio <= higher_bound:
            filtered.append(box)
    return filtered



def merge_char_words(boxes, x_thresh=20, y_thresh=10):
    used = [False] * len(boxes)
    words = []

    for i in range(len(boxes)):
        if used[i]:
            continue

        word = [boxes[i]]
        used[i] = True

        changed = True

        while changed:
            changed = False

            for j in range(len(boxes)):
                if used[j]:
                    continue

                x1, y1, w1, h1 = boxes[j]

                for wx, wy, ww, wh in word:

                    if abs(wy - y1) < y_thresh:

                        if abs((wx + ww) - x1) < x_thresh or abs((x1 + w1) - wx) < x_thresh:
                            word.append(boxes[j])
                            used[j] = True
                            changed = True
                            break

        words.append(sorted(word, key=lambda b: b[0]))

    return words


def get_word_boxes(words):
    word_boxes = []

    for word in words:
        x_min = min(x for x, y, w, h in word)
        y_min = min(y for x, y, w, h in word)
        x_max = max(x + w for x, y, w, h in word)
        y_max = max(y + h for x, y, w, h in word)

        word_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return word_boxes



def sort_boxes_reading_order(boxes, y_thresh=10):

    boxes = sorted(boxes, key=lambda b: b[1])

    lines = []

    for box in boxes:
        x, y, w, h = box
        placed = False

        for line in lines:
            if abs(line[0][1] - y) < y_thresh:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])


    for line in lines:
        line.sort(key=lambda b: b[0])


    return [b for line in lines for b in line]

def sort_word_chars(boxes): 
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def remove_image_border_box(boxes, image_shape):
    img_h, img_w = image_shape[:2]
    filtered_boxes = []
    for box in boxes:
        _, _, w, h = box
        box_area = w * h
        image_area = img_w * img_h
        if box_area < 0.8 * image_area: 
            filtered_boxes.append(box)
    return filtered_boxes

def remove_holes(boxes):
    filtered = []

    for i, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        is_inside = False

        for j, box2 in enumerate(boxes):
            if i == j:
                continue

            x2, y2, w2, h2 = box2

            if (x2 <= x1 and y2 <= y1 and
                x2 + w2 >= x1 + w1 and
                y2 + h2 >= y1 + h1):
                is_inside = True
                break

        if not is_inside:
            filtered.append(box1)

    return filtered


def remove_large_boxes(boxes, image_shape, ratio=0.5):
    filtered_boxes = []
    for box in boxes:
        x, y, w, h = box
        if w < ratio * image_shape[1]:
            filtered_boxes.append(box)
    return filtered_boxes