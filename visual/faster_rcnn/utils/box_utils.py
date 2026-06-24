import torch

def bbox_transform(anchors, gt):
    wa = anchors[:,2] - anchors[:,0]
    ha = anchors[:,3] - anchors[:,1]
    xa = anchors[:,0] + .5*wa
    ya = anchors[:,1] + .5*ha
    w = gt[:,2] - gt[:,0]
    h = gt[:,3] - gt[:,1]
    x = gt[:,0] + .5*w
    y = gt[:,1] + .5*h
    tx = (x - xa) / (wa + 1e-8)
    ty = (y - ya) / (ha + 1e-8)
    tw = torch.log(w / (wa + 1e-8) + 1e-8)
    th = torch.log(h / (ha + 1e-8) + 1e-8)
    return torch.stack([tx, ty, tw, th], 1)

def bbox_transform_inv(anchors, deltas):
    wa = anchors[:,2] - anchors[:,0]
    ha = anchors[:,3] - anchors[:,1]
    xa = anchors[:,0] + .5*wa
    ya = anchors[:,1] + .5*ha
    tx, ty = deltas[:,0], deltas[:,1]
    tw = deltas[:,2].clamp(max=4.)
    th = deltas[:,3].clamp(max=4.)
    x = tx*wa + xa
    y = ty*ha + ya
    w = torch.exp(tw)*wa
    h = torch.exp(th)*ha
    return torch.stack([x-.5*w, y-.5*h, x+.5*w, y+.5*h], 1)



def iou_matrix(a, b):
    # a: (N,4), b: (M,4) -> (N,M)
    # compare each box in a to each box in b, return IoU
    ix1 = torch.max(a[:,0].unsqueeze(1), b[:,0].unsqueeze(0))
    iy1 = torch.max(a[:,1].unsqueeze(1), b[:,1].unsqueeze(0))
    ix2 = torch.min(a[:,2].unsqueeze(1), b[:,2].unsqueeze(0))
    iy2 = torch.min(a[:,3].unsqueeze(1), b[:,3].unsqueeze(0))
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    area_a = ((a[:,2]-a[:,0])*(a[:,3]-a[:,1])).unsqueeze(1)
    area_b = ((b[:,2]-b[:,0])*(b[:,3]-b[:,1])).unsqueeze(0)
    return inter / (area_a + area_b - inter + 1e-8)


def clip_boxes(boxes, img_shape):
    H, W = img_shape
    boxes[:,0::2] = boxes[:,0::2].clamp(0, W)
    boxes[:,1::2] = boxes[:,1::2].clamp(0, H)
    return boxes
def filter_boxes(boxes, min_size):
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    return (w >= min_size) & (h >= min_size)


def nms(boxes, scores, thresh):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        ious = iou_matrix(boxes[i].unsqueeze(0), boxes[rest])[0]
        order = rest[ious <= thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
