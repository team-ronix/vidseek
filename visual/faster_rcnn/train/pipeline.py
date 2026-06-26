import argparse, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from visual.faster_rcnn.voc_dataset import VOCDataset, VOC_CLASSES, collate_fn
from visual.faster_rcnn.utils.eval import VOCEvaluator
from visual.faster_rcnn.train.steps import step1, step2, step3, step4

def build_dataset(data_path):
    TRAIN_VAL_ROOT = data_path + '/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    TEST_ROOT = data_path + '/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
    trainval_ds = VOCDataset(base=TRAIN_VAL_ROOT, split='trainval')
    test_ds = VOCDataset(base=TEST_ROOT, split='test')
    print(f'\nFinal Count -> Trainval: {len(trainval_ds)}, Test: {len(test_ds)}')
    trainval_loader = DataLoader(
        trainval_ds, batch_size=1, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    return trainval_loader, test_loader

def run_eval(model, loader, desc, num_classes, device):
    model.eval()
    evaluator = VOCEvaluator(num_classes=num_classes, iou_thresh=0.5)
    with torch.no_grad():
        for imgs, shapes, gt_bl, gt_ll, gt_dl, ids in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            res = model(imgs, shapes)[0]
            evaluator.update(
                ids[0],
                res['boxes'].cpu(), res['scores'].cpu(), res['labels'].cpu(),
                gt_bl[0].cpu(), gt_ll[0].cpu(),
                gt_difficult=gt_dl[0].cpu()
            )
    mAP, aps = evaluator.compute_map()
    print(f'\n{desc} - mAP @ IoU=0.5 : {mAP*100:.1f}%')
    print(f"  {'Class':<16} {'AP':>6}")
    print(f"  {'-'*24}")
    for cls, ap in sorted(aps.items(), key=lambda x: -x[1]):
        print(f"  {cls:<16s} {ap*100:5.1f}%")
    return mAP, aps

def save_chart(mAP, aps, save_path):
    colors = [
        '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231',
        '#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4',
        '#469990','#dcbeff','#9A6324','#fffac8','#800000',
        '#aaffc3','#808000','#ffd8b1','#000075','#a9a9a9',
    ]
    sorted_items = sorted(aps.items(), key=lambda x: -x[1])
    cls_names = [k for k, _ in sorted_items]
    ap_vals = [v * 100 for _, v in sorted_items]
    colors = [colors[list(VOC_CLASSES).index(c) % 20] for c in cls_names]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(cls_names, ap_vals, color=colors, edgecolor='white', height=0.7)
    ax.axvline(mAP * 100, color='black', linestyle='--', linewidth=1.5,
            label=f'mAP = {mAP*100:.1f}%')

    for bar, val in zip(bars, ap_vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=8)

    ax.set_xlabel('Average Precision (%)', fontsize=11)
    ax.set_title(f'Per-Class AP @ IoU=0.5  |  mAP = {mAP*100:.1f}%', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to the VOC dataset")
    parser.add_argument("--iter-warm", help="Number of warm-up iterations", type=int, default=60_000)
    parser.add_argument("--iter-cool", help="Number of cool-down iterations", type=int, default=20_000)
    parser.add_argument("--iter-ft-warm", help="Number of fitting warm-up iterations", type=int, default=20_000)
    parser.add_argument("--iter-ft-cool", help="Number of fitting cool-down iterations", type=int, default=10_000)
    parser.add_argument("--checkpoint-path", help="Path to the checkpoint file", type=str, default='checkpoints')
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {DEVICE}')
    os.makedirs(args.checkpoint_path, exist_ok=True)
    NUM_CLASSES = len(VOC_CLASSES)
    trainval_loader, test_loader = build_dataset(args.data_path)
    model_s1 = step1(args.checkpoint_path, trainval_loader, NUM_CLASSES, DEVICE, args.iter_warm, args.iter_cool)
    model_s2 = step2(model_s1, args.checkpoint_path, trainval_loader, NUM_CLASSES, DEVICE, args.iter_warm, args.iter_cool)
    model_s3 = step3(model_s1, model_s2, args.checkpoint_path, trainval_loader, NUM_CLASSES, DEVICE, args.iter_ft_warm, args.iter_ft_cool)
    model_final = step4(model_s2, model_s3, args.checkpoint_path, trainval_loader, NUM_CLASSES, DEVICE, args.iter_ft_warm, args.iter_ft_cool)
    mAP, aps = run_eval(model_final, test_loader, "Final Model", NUM_CLASSES, DEVICE)
    save_chart(mAP, aps, f"{args.checkpoint_path}/final_chart.png")
