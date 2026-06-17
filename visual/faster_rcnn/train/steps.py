import os
import torch
from model.faster_rcnn import FasterRCNN
from visual.faster_rcnn.train.learning import train_rpn_iters, train_det_iters, clear_mid_ckpts


def _freeze_conv_layers(model):
    for p in model.backbone.block1.parameters():
        p.requires_grad = False
    for p in model.backbone.block2.parameters():
        p.requires_grad = False

def step1(chck_path, data_loader, num_classes, device, iter_warm, iter_cool):
    step1_path = f'{chck_path}/step1_rpn.pth'
    if os.path.exists(step1_path):
        print('Checkpoint found - loading Step 1 weights')
        model_s1 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s1)
        model_s1.load_state_dict(torch.load(step1_path, map_location=device))
        model_s1.eval()
    else:
        model_s1 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s1)
        train_rpn_iters(model_s1, data_loader, device, iter_warm, 1e-3, 
                        'Step1-warm', ckpt_prefix=f'{chck_path}/step1_warm', ckpt_every=20_000)
        train_rpn_iters(model_s1, data_loader, device, iter_cool, 1e-4, 
                        'Step1-cool', ckpt_prefix=f'{chck_path}/step1_cool', ckpt_every=20_000)
        torch.save(model_s1.state_dict(), step1_path)
        clear_mid_ckpts(f'{chck_path}/step1_warm')
        clear_mid_ckpts(f'{chck_path}/step1_cool')
    return model_s1

def step2(model_s1, chck_path, data_loader, num_classes, device, iter_warm, iter_cool):
    step2_path = f'{chck_path}/step2_detector.pth'
    if os.path.exists(step2_path):
        print('Checkpoint found - loading Step 2 weights')
        model_s2 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s2)
        model_s2.load_state_dict(torch.load(step2_path, map_location=device))
        model_s2.eval()
    else:
        model_s2 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s2)
        train_det_iters(model_s2, model_s1, data_loader, device, iter_warm, 1e-3, 
                        'Step2-warm', ckpt_prefix=f'{chck_path}/step2_warm', ckpt_every=20_000)
        train_det_iters(model_s2, model_s1, data_loader, device, iter_cool, 1e-4, 
                        'Step2-cool', ckpt_prefix=f'{chck_path}/step2_cool', ckpt_every=20_000)
        torch.save(model_s2.state_dict(), step2_path)
        clear_mid_ckpts(f'{chck_path}/step2_warm')
        clear_mid_ckpts(f'{chck_path}/step2_cool')
    return model_s2

def step3(model_s1, model_s2, chck_path, data_loader, num_classes, device, iter_ft_warm, iter_ft_cool):
    step3_path = f'{chck_path}/step3_rpn_shared.pth'
    if os.path.exists(step3_path):
        print('Checkpoint found - loading Step 3 weights')
        model_s3 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s3)
        model_s3.load_state_dict(torch.load(step3_path, map_location=device))
        model_s3.eval()
    else:
        model_s3 = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_s3)
        model_s3.backbone.load_state_dict(model_s2.backbone.state_dict())
        model_s3.rpn.load_state_dict(model_s1.rpn.state_dict())
        for p in model_s3.backbone.parameters():
            p.requires_grad = False

        train_rpn_iters(model_s3, data_loader, device, iter_ft_warm, 1e-3,
                        'Step3-warm', ckpt_prefix=f'{chck_path}/step3_warm', ckpt_every=20_000)
        train_rpn_iters(model_s3, data_loader, device, iter_ft_cool, 1e-4,
                        'Step3-cool', ckpt_prefix=f'{chck_path}/step3_cool', ckpt_every=10_000)

        torch.save(model_s3.state_dict(), step3_path)
        clear_mid_ckpts(f'{chck_path}/step3_warm')
        clear_mid_ckpts(f'{chck_path}/step3_cool')
    return model_s3


def step4(model_s2, model_s3, chck_path, data_loader, num_classes, device, iter_ft_warm, iter_ft_cool):
    final_path = f'{chck_path}/faster_rcnn_final.pth'
    if os.path.exists(final_path):
        print('Checkpoint found - loading Step 4 weights')
        model_final = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_final)
        model_final.load_state_dict(torch.load(final_path, map_location=device))
        model_final.eval()
    else:
        model_final = FasterRCNN(num_classes=num_classes).to(device)
        _freeze_conv_layers(model_final)
        model_final.backbone.load_state_dict(model_s3.backbone.state_dict())
        model_final.rpn.load_state_dict(model_s3.rpn.state_dict())
        model_final.det_head.load_state_dict(model_s2.det_head.state_dict())

        for p in model_final.backbone.parameters():
            p.requires_grad = False
        for p in model_final.rpn.parameters():
            p.requires_grad = False

        train_det_iters(model_final, model_final, data_loader, device, iter_ft_warm, 1e-3,
                        'Step4-warm', ckpt_prefix=f'{chck_path}/step4_warm', ckpt_every=20_000)
        train_det_iters(model_final, model_final, data_loader, device, iter_ft_cool, 1e-4,
                        'Step4-cool', ckpt_prefix=f'{chck_path}/step4_cool', ckpt_every=10_000)

        torch.save(model_final.state_dict(), final_path)
        print(f'Final model saved -> {final_path}')
        clear_mid_ckpts(f'{chck_path}/step4_warm')
        clear_mid_ckpts(f'{chck_path}/step4_cool')
    return model_final