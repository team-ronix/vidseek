import numpy as np
from collections import defaultdict
from visual.vrd_ml.vrd_dataset import BBox


def _triplet_match(pred_triplet, gt_triplets, iou_thresh=0.5, box_match=True):
    ps_lbl, pp_lbl, po_lbl = pred_triplet[0], pred_triplet[1], pred_triplet[2]
    ps_box = BBox(*pred_triplet[3]) if len(pred_triplet) > 3 else None
    po_box = BBox(*pred_triplet[4]) if len(pred_triplet) > 4 else None
    for gt in gt_triplets:
        gs_lbl, gp_lbl, go_lbl = gt[0], gt[1], gt[2]
        if ps_lbl != gs_lbl or pp_lbl != gp_lbl or po_lbl != go_lbl:
            continue
        if not box_match:
            return True
        gs_box = BBox(*gt[3]) if len(gt) > 3 else None
        go_box = BBox(*gt[4]) if len(gt) > 4 else None
        if gs_box and ps_box and go_box and po_box:
            if ps_box.iou(gs_box) >= iou_thresh and po_box.iou(go_box) >= iou_thresh:
                return True
    return False


def recall_at_k(predictions, ground_truths, k=50, iou_thresh=0.5, box_match=False):
    total_gt = 0
    total_hit = 0
    for preds, gts in zip(predictions, ground_truths):
        if not gts:
            continue
        top_preds = preds[:k]
        matched = set()
        for p in top_preds:
            p_tuple = (p["subj"], p["pred"], p["obj"])
            if box_match == True:
                p_tuple += (tuple(p.get("subj_box", [])), tuple(p.get("obj_box", [])))
            for gi, g in enumerate(gts):
                if gi in matched:
                    continue
                g_tuple = (g["subj"], g["pred"], g["obj"])
                if box_match == True:
                    g_tuple += (tuple(g.get("subj_box", [])), tuple(g.get("obj_box", [])))
                if _triplet_match(p_tuple, [g_tuple], iou_thresh, box_match):
                    matched.add(gi)
                    break
        total_gt += len(gts)
        total_hit += len(matched)
    return total_hit / total_gt if total_gt > 0 else 0.0


def mean_recall_at_k(predictions, ground_truths, k=50, iou_thresh=0.5, box_match=False):
    pred_hits = defaultdict(int)
    pred_total = defaultdict(int)
    for preds, gts in zip(predictions, ground_truths):
        if not gts:
            continue
        top_preds = preds[:k]
        matched = set()
        for p in top_preds:
            p_tuple = (p["subj"], p["pred"], p["obj"])
            for gi, g in enumerate(gts):
                if gi in matched:
                    continue
                g_tuple = (g["subj"], g["pred"], g["obj"])
                if _triplet_match(p_tuple, [g_tuple], iou_thresh, box_match):
                    matched.add(gi)
                    pred_hits[g["pred"]] += 1
                    break
        # count gt per predicate
        for g in gts:
            pred_total[g["pred"]] += 1
    # compute per-predicate recall
    per_pred = {}
    for pred, total in pred_total.items():
        per_pred[pred] = pred_hits.get(pred, 0) / total
    overall = float(np.mean(list(per_pred.values()))) if per_pred else 0.0
    return overall, per_pred


def zero_shot_recall_at_k(predictions, ground_truths, seen_triplets, k=50):
    zs_preds = []
    zs_gts = []
    for preds, gts in zip(predictions, ground_truths):
        # filter out gt triplets that were in the training set
        zs_gt = [g for g in gts
                 if (g["subj"], g["pred"], g["obj"]) not in seen_triplets]
        if not zs_gt:
            continue
        zs_preds.append(preds)
        zs_gts.append(zs_gt)
    return recall_at_k(zs_preds, zs_gts, k=k, box_match=False)


def evaluation_report(predictions, ground_truths, seen_triplets=None):
    r50 = recall_at_k(predictions, ground_truths, k=50)
    r100 = recall_at_k(predictions, ground_truths, k=100)
    mr50, per50 = mean_recall_at_k(predictions, ground_truths, k=50)
    mr100, per100 = mean_recall_at_k(predictions, ground_truths, k=100)
    report = {
        "Recall@50": round(r50 * 100, 2),
        "Recall@100": round(r100 * 100, 2),
        "mRecall@50": round(mr50 * 100, 2),
        "mRecall@100": round(mr100 * 100, 2),
        "per_pred_R@50": {k: round(v * 100, 2) for k, v in per50.items()},
        "per_pred_R@100": {k: round(v * 100, 2) for k, v in per100.items()},
    }
    if seen_triplets != None:
        zs50 = zero_shot_recall_at_k(predictions, ground_truths, seen_triplets, k=50)
        zs100 = zero_shot_recall_at_k(predictions, ground_truths, seen_triplets, k=100)
        report["ZeroShot_Recall@50"] = round(zs50 * 100, 2)
        report["ZeroShot_Recall@100"] = round(zs100 * 100, 2)
    return report


def print_report(report):
    print("\n" + "=" * 48)
    print("  VRD Evaluation Report")
    print("=" * 48)
    keys = ["Recall@50", "Recall@100", "mRecall@50", "mRecall@100",
            "ZeroShot_Recall@50", "ZeroShot_Recall@100"]
    for k in keys:
        if k in report:
            print(f"  {k:<26} {report[k]:>6.2f} %")
    print("=" * 48)
    if "per_pred_R@50" in report:
        print("\n  Top-10 predicates by R@50:")
        per = report["per_pred_R@50"]
        top = sorted(per.items(), key=lambda x: -x[1])[:10]
        for pred, val in top:
            bar = "█" * int(val / 5)
            print(f"  {pred:<22} {val:>5.1f}%  {bar}")
    print()
