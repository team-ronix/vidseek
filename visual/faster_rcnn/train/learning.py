import glob, re, os
import torch
import torch.optim as optim
from tqdm import tqdm

def make_sgd(params, lr=1e-3):
    return optim.SGD(
        [p for p in params if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4
    )

def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr

def _latest_mid_ckpt(ckpt_prefix):
    pattern = ckpt_prefix + '_mid_iter*.pth'
    candidates = glob.glob(pattern)
    best_path, best_iter = None, 0
    for p in candidates:
        m = re.search(r'_mid_iter(\d+)\.pth$', p)
        if m:
            n = int(m.group(1))
            if n > best_iter:
                best_iter, best_path = n, p
    return best_path, best_iter

def clear_mid_ckpts(ckpt_prefix):
    pattern = ckpt_prefix + '_mid_iter*.pth'
    candidates = glob.glob(pattern)
    for p in candidates:
        os.remove(p)
        print(f' Deleted mid-ckpt {p}')

def train_rpn_iters(model, loader, device, n_iters, lr, step_name, ckpt_prefix=None, ckpt_every=30_000):
    params = list(model.backbone.parameters()) + list(model.rpn.parameters())
    opt = make_sgd(params, lr)
    model.train()
    start_iter = 0
    if ckpt_prefix:
        mid_path, mid_iter = _latest_mid_ckpt(ckpt_prefix)
        if mid_path and mid_iter < n_iters:
            print(f'Resuming {step_name} from mid-ckpt iter {mid_iter:,} ({mid_path})')
            model.load_state_dict(torch.load(mid_path, map_location=device))
            start_iter = mid_iter
        elif mid_path and mid_iter >= n_iters:
            print(f'Mid-ckpt at iter {mid_iter:,} already covers {n_iters:,} iters - skipping.')
            model.load_state_dict(torch.load(mid_path, map_location=device))
            return

    done = start_iter
    pbar = tqdm(total=n_iters, initial=done, desc=f'{step_name} lr={lr}', unit='iter')
    while done < n_iters:
        for imgs, shapes, gt_bl, _, _, _ in loader:
            if done >= n_iters:
                break
            imgs = imgs.to(device)
            gt_b = [b.to(device) for b in gt_bl]
            feat = model.backbone(imgs)
            _, rc, rr = model.rpn(feat, shapes[0], gt_b[0])
            if rc is None:
                continue
            loss = rc + rr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            done += 1
            pbar.set_postfix(rpn_cls=f'{rc.item():.3f}', rpn_reg=f'{rr.item():.3f}')
            pbar.update(1)
            if ckpt_prefix and done % ckpt_every == 0 and done <= n_iters:
                mid_path = f'{ckpt_prefix}_mid_iter{done}.pth'
                torch.save(model.state_dict(), mid_path)
                print(f'[ckpt] saved {mid_path}')
    pbar.close()


def train_det_iters(model, rpn_src, loader, device, n_iters, lr, step_name, ckpt_prefix=None, ckpt_every=30_000):
    same_model = model is rpn_src
    params = list(model.backbone.parameters()) + list(model.det_head.parameters())
    opt = make_sgd(params, lr)
    if not same_model:
        rpn_src.eval()
    start_iter = 0
    if ckpt_prefix:
        mid_path, mid_iter = _latest_mid_ckpt(ckpt_prefix)
        if mid_path and mid_iter < n_iters:
            print(f'Resuming {step_name} from mid-ckpt iter {mid_iter:,} ({mid_path})')
            model.load_state_dict(torch.load(mid_path, map_location=device))
            start_iter = mid_iter
        elif mid_path and mid_iter >= n_iters:
            print(f'Mid-ckpt at iter {mid_iter:,} already covers {n_iters:,} iters - skipping.')
            model.load_state_dict(torch.load(mid_path, map_location=device))
            return

    done = start_iter
    pbar = tqdm(total=n_iters, initial=done, desc=f'{step_name} lr={lr}', unit='iter')
    while done < n_iters:
        for imgs, shapes, gt_bl, gt_ll, _, _ in loader:
            if done >= n_iters:
                break
            imgs = imgs.to(device)
            gt_b = [b.to(device) for b in gt_bl]
            gt_l = [l.to(device) for l in gt_ll]
            if same_model:
                rpn_src.eval()
            with torch.no_grad():
                fr = rpn_src.backbone(imgs)
                props, _, _ = rpn_src.rpn(fr, shapes[0])
            model.det_head.train()
            if any(p.requires_grad for p in model.backbone.parameters()):
                model.backbone.train()
                feat = model.backbone(imgs)
            else:
                feat = fr
            dc, dr = model.det_head(
                feat, props.detach(), gt_b[0], gt_l[0]
            )
            loss = dc + dr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            done += 1
            pbar.set_postfix(det_cls=f'{dc.item():.3f}', det_reg=f'{dr.item():.3f}')
            pbar.update(1)
            if ckpt_prefix and done % ckpt_every == 0 and done <= n_iters:
                mid_path = f'{ckpt_prefix}_mid_iter{done}.pth'
                torch.save(model.state_dict(), mid_path)
                print(f'[ckpt] saved {mid_path}')
    pbar.close()