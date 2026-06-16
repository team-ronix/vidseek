# Import required modules
import cv2
import math
import numpy as np

############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]




def extract_word_images(image, net, pad=20, min_area_ratio=0.001,
                        inp_size=320, conf_thresh=0.5, nms_thresh=0.4, score_thresh=0.8):
    H, W = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (inp_size, inp_size),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    boxes, confidences = decode(scores, geometry, score_thresh)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_thresh, nms_thresh)

    rW, rH = W / inp_size, H / inp_size
    word_images = []

    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        center, (w, h), angle = boxes[idx]

        bw = int(w * rW)
        bh = int(h * rH)

        if (bw * bh) / (W * H) < min_area_ratio:
            continue

        cx = int(center[0] * rW)
        cy = int(center[1] * rH)

        rect = ((cx, cy), (bw + pad * 2, bh + pad * 2), angle)
        pts = cv2.boxPoints(rect)
        x1 = max(int(pts[:, 0].min()), 0)
        y1 = max(int(pts[:, 1].min()), 0)
        x2 = min(int(pts[:, 0].max()), W)
        y2 = min(int(pts[:, 1].max()), H)

        word_images.append(image[y1:y2, x1:x2])

    return word_images