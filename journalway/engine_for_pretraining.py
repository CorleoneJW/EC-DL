# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

import utils
from einops import rearrange
import torch.nn.functional as F
from masking_generator import RandomMaskingGenerator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import lpips

def log_max_memory_usage():
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Max Memory Allocated: {max_memory_allocated / 1024**2:.2f} MB")
    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计，为下一次测量做准备

def similarityLoss(tensor1, tensor2):
    # 归一化输入张量
    tensor1_normalized = F.normalize(tensor1, p=2, dim=2)
    tensor2_normalized = F.normalize(tensor2, p=2, dim=2)

    # 计算余弦相似度
    cosine = F.cosine_similarity(tensor1_normalized, tensor2_normalized, dim=2)

    loss = 1 - cosine.mean()
    return loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            # outputs, _ = model(images, bool_masked_pos)        # for the moe and other model
            outputs = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_withfeature(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs,encoder_feature = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output
def train_one_epoch_meanTeacher(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, single_layer_MLP=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func_A = nn.MSELoss()
    loss_func_B = nn.MSELoss()
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A = model(images, bool_masked_pos_A)    #torch.Size([bs, 147, 768])
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # encoder_linear_A = single_layer_MLP(encoder_feature_A)
            with torch.no_grad():
                outputs_B = model(images, bool_masked_pos_B)    #torch.Size([bs, 147, 768])
                # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
                # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
                # encoder_linear_B = single_layer_MLP(encoder_feature_B)
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # loss_align = similarityLoss(encoder_linear_A,encoder_linear_B)
            loss_align = loss_func_B(input=outputs_A, target=outputs_B)
        loss = loss_decoder + 0.5*loss_align
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, loss_decoder.item(), loss_align.item()

# doing alignment on the output of encoder
def train_one_epoch_feature_alignment_pre(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    loss_func_B = nn.MSELoss()
    # loss_fn_vgg = lpips.LPIPS(net='vgg')
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            with torch.no_grad():
                outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
                # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
                # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate loss B (CLIP loss)
            encoder_feature_A_N = F.normalize(encoder_feature_A, p=2, dim=-1)
            encoder_feature_B_N = F.normalize(encoder_feature_B, p=2, dim=-1)
            encoder_feature_B_N_trans = encoder_feature_B_N.transpose(1,2)
            logits = torch.bmm(encoder_feature_A_N, encoder_feature_B_N_trans)
            # logits_N = F.normalize(logits, p=2, dim=-1)
            # logitsT = torch.transpose(logits, dim0=1,dim1=2)
            # logitsT_N = F.normalize(logitsT, p=2, dim=-1)
            unmasked_pathches = int(window_size[0]*window_size[1]*(1-mask_ratio))
            wish_labels = torch.arange(unmasked_pathches).to(device)       # to make the diagonal approximate to 1 , 49 is the patchd number of unmasked part
            wish_labels = wish_labels.unsqueeze(0).expand(data_loader.batch_size, unmasked_pathches)       # shape: (bs,49)
            one_hot_labels = F.one_hot(wish_labels, num_classes=unmasked_pathches).float()
            loss_align = loss_func_B(input=logits, target=one_hot_labels)
        weighted_loss_align = 5 * loss_align
        loss = loss_decoder + weighted_loss_align
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())
        metric_logger.update(loss_align=weighted_loss_align.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# doing alignment on the output of encoder with ema
def train_one_epoch_feature_alignment_ema(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, ema = None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func_A = nn.MSELoss()
    loss_func_B = nn.MSELoss()
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            with torch.no_grad():
                with ema.average_parameters():
                    outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
                    # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
                    # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate loss B (CLIP loss)
            encoder_feature_A_N = F.normalize(encoder_feature_A, p=2, dim=-1)
            encoder_feature_B_N = F.normalize(encoder_feature_B, p=2, dim=-1)
            encoder_feature_B_N_trans = encoder_feature_B_N.transpose(1,2)
            logits = torch.bmm(encoder_feature_A_N, encoder_feature_B_N_trans)
            # logits_N = F.normalize(logits, p=2, dim=-1)
            # logitsT = torch.transpose(logits, dim0=1,dim1=2)
            # logitsT_N = F.normalize(logitsT, p=2, dim=-1)
            unmasked_pathches = int(window_size[0]*window_size[1]*(1-mask_ratio))
            wish_labels = torch.arange(unmasked_pathches).to(device)       # to make the diagonal approximate to 1 , 49 is the patchd number of unmasked part
            wish_labels = wish_labels.unsqueeze(0).expand(data_loader.batch_size, unmasked_pathches)       # shape: (bs,49)
            one_hot_labels = F.one_hot(wish_labels, num_classes=unmasked_pathches).float()
            loss_align = loss_func_B(input=logits, target=one_hot_labels)
        weighted_loss_align = 5 * loss_align
        loss = loss_decoder + weighted_loss_align
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # call the loss_scaler to update the parameters of model
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # update the ema
        ema.update()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())
        metric_logger.update(loss_align=weighted_loss_align.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output of encoder, add perceptual loss
def train_one_epoch_feature_alignment(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    loss_func_B = nn.MSELoss()
    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
            # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
            # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)

            # calculate loss B (CLIP loss)
            encoder_feature_A_N = F.normalize(encoder_feature_A, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_B_N = F.normalize(encoder_feature_B, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_A_mean = encoder_feature_A_N.mean(dim=1)        # shape: [bs,768]
            encoder_feature_B_mean = encoder_feature_B_N.mean(dim=1)        # shape: [bs,768]
            """
            ####  previous FA in feature dimension  ####
            encoder_feature_B_N_trans = encoder_feature_B_N.transpose(1,2)
            logits = torch.bmm(encoder_feature_A_N, encoder_feature_B_N_trans)
            logits_N = F.normalize(logits, p=2, dim=-1)
            logitsT = torch.transpose(logits, dim0=1,dim1=2)
            ogitsT_N = F.normalize(logitsT, p=2, dim=-1)
            unmasked_pathches = int(window_size[0]*window_size[1]*(1-mask_ratio))
            wish_labels = torch.arange(unmasked_pathches).to(device)       # to make the diagonal approximate to 1 , 49 is the patchd number of unmasked part
            wish_labels = wish_labels.unsqueeze(0).expand(data_loader.batch_size, unmasked_pathches)       # shape: (bs,49)
            one_hot_labels = F.one_hot(wish_labels, num_classes=unmasked_pathches).float()
            """
            encoder_feature_A_mean_norm = encoder_feature_A_mean.norm(dim=1,keepdim=True)
            encoder_feature_B_mean_norm = encoder_feature_B_mean.norm(dim=1,keepdim=True)
            logits = (torch.matmul(encoder_feature_A_mean, encoder_feature_B_mean.t())) / (torch.matmul(encoder_feature_A_mean_norm, encoder_feature_B_mean_norm.t()))
            logits = (logits+1)/2       # range: from [-1,1] to [0,1], shape: [bs,bs]
            one_hot_labels = torch.full((data_loader.batch_size, data_loader.batch_size), 0.0002).fill_diagonal_(1).float().to(device)
            loss_align = loss_func_B(input=logits, target=one_hot_labels)
        weighted_loss_align = 5 * loss_align
        loss = loss_decoder + weighted_loss_align
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())
        metric_logger.update(loss_align=weighted_loss_align.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output of encoder, add perceptual loss
def train_one_epoch_feature_alignment_pl(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    loss_func_B = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
            # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
            # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate perceptional loss
            with torch.no_grad():
                reconstuct_images_patch = images_patch.clone()
                batch_size, num_patches, dim = reconstuct_images_patch.shape
                for i in range(batch_size):
                    masked_indices = bool_masked_pos_A[i]  # 当前批次的掩码
                    reconstuct_images_patch[i, masked_indices] = outputs_A[i].float()
                # reconstuct_images_patch[bool_masked_pos_A] = outputs_A.float()
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_pre = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_pre = rec_img_pre * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_pre = rearrange(rec_img_pre, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_pre = rec_img_pre[0, :].clip(0,0.996)
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_after = rearrange(reconstuct_images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_after = rec_img_after * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_after = rearrange(rec_img_after, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_after = rec_img_after[0, :].clip(0,0.996)
                # from [0,1] normalize tensor to [-1,1]
                min_val_after = torch.min(rec_img_after)
                max_val_after = torch.max(rec_img_after)
                min_val_pre = torch.min(rec_img_pre)
                max_val_pre = torch.max(rec_img_pre)
                rec_img_after = 2 * (rec_img_after - min_val_after) / (max_val_after - min_val_after) - 1
                rec_img_pre = 2 * (rec_img_pre - min_val_pre) / (max_val_pre - min_val_pre) - 1
                rec_img_after = torch.unsqueeze(rec_img_after, dim=0)       # shape: [1,3,224,224]
                rec_img_pre = torch.unsqueeze(rec_img_pre, dim=0)
            # calculate the perceptional loss
            loss_pl = loss_fn_vgg(rec_img_after.to(device), rec_img_pre.to(device))     # shape: [1,3,224,224]
            # calculate loss B (CLIP loss)
            encoder_feature_A_N = F.normalize(encoder_feature_A, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_B_N = F.normalize(encoder_feature_B, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_A_mean = encoder_feature_A_N.mean(dim=1)        # shape: [bs,768]
            encoder_feature_B_mean = encoder_feature_B_N.mean(dim=1)        # shape: [bs,768]
            """
            ####  previous FA in feature dimension  ####
            encoder_feature_B_N_trans = encoder_feature_B_N.transpose(1,2)
            logits = torch.bmm(encoder_feature_A_N, encoder_feature_B_N_trans)
            logits_N = F.normalize(logits, p=2, dim=-1)
            logitsT = torch.transpose(logits, dim0=1,dim1=2)
            ogitsT_N = F.normalize(logitsT, p=2, dim=-1)
            unmasked_pathches = int(window_size[0]*window_size[1]*(1-mask_ratio))
            wish_labels = torch.arange(unmasked_pathches).to(device)       # to make the diagonal approximate to 1 , 49 is the patchd number of unmasked part
            wish_labels = wish_labels.unsqueeze(0).expand(data_loader.batch_size, unmasked_pathches)       # shape: (bs,49)
            one_hot_labels = F.one_hot(wish_labels, num_classes=unmasked_pathches).float()
            """
            encoder_feature_A_mean_norm = encoder_feature_A_mean.norm(dim=1,keepdim=True)
            encoder_feature_B_mean_norm = encoder_feature_B_mean.norm(dim=1,keepdim=True)
            logits = (torch.matmul(encoder_feature_A_mean, encoder_feature_B_mean.t())) / (torch.matmul(encoder_feature_A_mean_norm, encoder_feature_B_mean_norm.t()))
            logits = (logits+1)/2       # range: from [-1,1] to [0,1], shape: [bs,bs]
            one_hot_labels = torch.full((data_loader.batch_size, data_loader.batch_size), 0.0002).fill_diagonal_(1).float().to(device)
            loss_align = loss_func_B(input=logits, target=one_hot_labels)
        weighted_loss_align = 5 * loss_align
        # loss_pl = loss_pl.item() 
        loss_pl = loss_pl.squeeze()
        loss = loss_decoder + weighted_loss_align + loss_pl
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_pl=loss_pl)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())
        metric_logger.update(loss_align=weighted_loss_align.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output of encoder, add perceptual loss
def train_one_epoch_feature_alignment_pl_stage2(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    loss_func_B = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, target, path) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
            # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
            # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate perceptional loss
            with torch.no_grad():
                reconstuct_images_patch = images_patch.clone()
                batch_size, num_patches, dim = reconstuct_images_patch.shape
                for i in range(batch_size):
                    masked_indices = bool_masked_pos_A[i]  # 当前批次的掩码
                    reconstuct_images_patch[i, masked_indices] = outputs_A[i].float()
                # reconstuct_images_patch[bool_masked_pos_A] = outputs_A.float()
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_pre = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_pre = rec_img_pre * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_pre = rearrange(rec_img_pre, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_pre = rec_img_pre[0, :].clip(0,0.996)
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_after = rearrange(reconstuct_images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_after = rec_img_after * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_after = rearrange(rec_img_after, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_after = rec_img_after[0, :].clip(0,0.996)
                # from [0,1] normalize tensor to [-1,1]
                min_val_after = torch.min(rec_img_after)
                max_val_after = torch.max(rec_img_after)
                min_val_pre = torch.min(rec_img_pre)
                max_val_pre = torch.max(rec_img_pre)
                rec_img_after = 2 * (rec_img_after - min_val_after) / (max_val_after - min_val_after) - 1
                rec_img_pre = 2 * (rec_img_pre - min_val_pre) / (max_val_pre - min_val_pre) - 1
                rec_img_after = torch.unsqueeze(rec_img_after, dim=0)       # shape: [1,3,224,224]
                rec_img_pre = torch.unsqueeze(rec_img_pre, dim=0)
            # calculate the perceptional loss
            loss_pl = loss_fn_vgg(rec_img_after.to(device), rec_img_pre.to(device))     # shape: [1,3,224,224]
            # calculate loss B (CLIP loss)
            encoder_feature_A_N = F.normalize(encoder_feature_A, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_B_N = F.normalize(encoder_feature_B, p=2, dim=-1)       # shape: [bs,49,768]
            encoder_feature_A_mean = encoder_feature_A_N.mean(dim=1)        # shape: [bs,768]
            encoder_feature_B_mean = encoder_feature_B_N.mean(dim=1)        # shape: [bs,768]
            """
            ####  previous FA in feature dimension  ####
            encoder_feature_B_N_trans = encoder_feature_B_N.transpose(1,2)
            logits = torch.bmm(encoder_feature_A_N, encoder_feature_B_N_trans)
            logits_N = F.normalize(logits, p=2, dim=-1)
            logitsT = torch.transpose(logits, dim0=1,dim1=2)
            ogitsT_N = F.normalize(logitsT, p=2, dim=-1)
            unmasked_pathches = int(window_size[0]*window_size[1]*(1-mask_ratio))
            wish_labels = torch.arange(unmasked_pathches).to(device)       # to make the diagonal approximate to 1 , 49 is the patchd number of unmasked part
            wish_labels = wish_labels.unsqueeze(0).expand(data_loader.batch_size, unmasked_pathches)       # shape: (bs,49)
            one_hot_labels = F.one_hot(wish_labels, num_classes=unmasked_pathches).float()
            """
            encoder_feature_A_mean_norm = encoder_feature_A_mean.norm(dim=1,keepdim=True)
            encoder_feature_B_mean_norm = encoder_feature_B_mean.norm(dim=1,keepdim=True)
            logits = (torch.matmul(encoder_feature_A_mean, encoder_feature_B_mean.t())) / (torch.matmul(encoder_feature_A_mean_norm, encoder_feature_B_mean_norm.t()))
            logits = (logits+1)/2       # range: from [-1,1] to [0,1], shape: [bs,bs]
            one_hot_labels = torch.full((data_loader.batch_size, data_loader.batch_size), 0.0002).fill_diagonal_(1).float().to(device)
            loss_align = loss_func_B(input=logits, target=one_hot_labels)
        weighted_loss_align = 5 * loss_align
        # loss_pl = loss_pl.item() 
        loss_pl = loss_pl.squeeze()
        loss = loss_decoder + weighted_loss_align + loss_pl
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_pl=loss_pl)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())
        metric_logger.update(loss_align=weighted_loss_align.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output of encoder, add perceptual loss
def train_one_epoch_pl(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    # loss_func_B = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
            # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
            # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate perceptional loss
            with torch.no_grad():
                reconstuct_images_patch = images_patch.clone()
                batch_size, num_patches, dim = reconstuct_images_patch.shape
                for i in range(batch_size):
                    masked_indices = bool_masked_pos_A[i]  # 当前批次的掩码
                    reconstuct_images_patch[i, masked_indices] = outputs_A[i].float()
                # reconstuct_images_patch[bool_masked_pos_A] = outputs_A.float()
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_pre = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_pre = rec_img_pre * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_pre = rearrange(rec_img_pre, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_pre = rec_img_pre[0, :].clip(0,0.996)
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_after = rearrange(reconstuct_images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_after = rec_img_after * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_after = rearrange(rec_img_after, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_after = rec_img_after[0, :].clip(0,0.996)
                # from [0,1] normalize tensor to [-1,1]
                min_val_after = torch.min(rec_img_after)
                max_val_after = torch.max(rec_img_after)
                min_val_pre = torch.min(rec_img_pre)
                max_val_pre = torch.max(rec_img_pre)
                rec_img_after = 2 * (rec_img_after - min_val_after) / (max_val_after - min_val_after) - 1
                rec_img_pre = 2 * (rec_img_pre - min_val_pre) / (max_val_pre - min_val_pre) - 1
                rec_img_after = torch.unsqueeze(rec_img_after, dim=0)       # shape: [1,3,224,224]
                rec_img_pre = torch.unsqueeze(rec_img_pre, dim=0)
            # calculate the perceptional loss
            loss_pl = loss_fn_vgg(rec_img_after.to(device), rec_img_pre.to(device))     # shape: [1,3,224,224]
            
        # loss_pl = loss_pl.item() 
        loss_pl = loss_pl.squeeze()
        loss = loss_decoder + loss_pl
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_pl=loss_pl)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# doing alignment on the output of encoder, add perceptual loss
def train_one_epoch_pl_stage2(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, window_size=None, mask_ratio=0.75, print_memory_usage=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # loss_func_A = nn.MSELoss()
    loss_func_A = nn.L1Loss()
    # loss_func_B = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    random_mask_generator = RandomMaskingGenerator(window_size,mask_ratio)

    for step, (batch, target, path) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos_A = batch   #iamges: torch.Size([bs,3,224,224]) bool_masked_pos_A: torch.Size([bs, 196])
        bool_masked_pos_B = []
        for i in range(data_loader.batch_size):
            temp_mask = random_mask_generator()
            temp_mask = torch.tensor(temp_mask)
            bool_masked_pos_B.append(temp_mask)
        bool_masked_pos_B = torch.stack(bool_masked_pos_B,dim=0)
        images = images.to(device, non_blocking=True)
        bool_masked_pos_A = bool_masked_pos_A.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        bool_masked_pos_B = bool_masked_pos_B.to(device, non_blocking=True).flatten(1).to(torch.bool)   # torch.Size([bs,196])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos_A].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs_A, encoder_feature_A = model(images, bool_masked_pos_A)    #outputs_A: torch.Size([bs, 147, 768]), encoder_feature_A: [bs,49,768]
            # encoder_feature_A = model.encoder.forward_features(images, bool_masked_pos_A)   #torch.Size([bs, 49, 768])
            # encoder_maxpool_A, _ = torch.topk(encoder_feature_A, k=10, dim=2)   #torch.Size([bs, 49, 10])
            outputs_B, encoder_feature_B = model(images, bool_masked_pos_B)    #ouputs_B: torch.Size([bs, 147, 768]), encoder_feature_B: [bs,49,768]
            # encoder_feature_B = model.encoder.forward_features(images, bool_masked_pos_B)   
            # encoder_maxpool_B, _ = torch.topk(encoder_feature_B, k=10, dim=2)   #torch.Size([bs, 49, 10])
            # calculate loss A
            loss_decoder = loss_func_A(input=outputs_A, target=labels)
            # calculate perceptional loss
            with torch.no_grad():
                reconstuct_images_patch = images_patch.clone()
                batch_size, num_patches, dim = reconstuct_images_patch.shape
                for i in range(batch_size):
                    masked_indices = bool_masked_pos_A[i]  # 当前批次的掩码
                    reconstuct_images_patch[i, masked_indices] = outputs_A[i].float()
                # reconstuct_images_patch[bool_masked_pos_A] = outputs_A.float()
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_pre = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_pre = rec_img_pre * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_pre = rearrange(rec_img_pre, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_pre = rec_img_pre[0, :].clip(0,0.996)
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img_after = rearrange(reconstuct_images_patch, 'b n (p c) -> b n p c', c=3)
                rec_img_after = rec_img_after * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                rec_img_after = rearrange(rec_img_after, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
                rec_img_after = rec_img_after[0, :].clip(0,0.996)
                # from [0,1] normalize tensor to [-1,1]
                min_val_after = torch.min(rec_img_after)
                max_val_after = torch.max(rec_img_after)
                min_val_pre = torch.min(rec_img_pre)
                max_val_pre = torch.max(rec_img_pre)
                rec_img_after = 2 * (rec_img_after - min_val_after) / (max_val_after - min_val_after) - 1
                rec_img_pre = 2 * (rec_img_pre - min_val_pre) / (max_val_pre - min_val_pre) - 1
                rec_img_after = torch.unsqueeze(rec_img_after, dim=0)       # shape: [1,3,224,224]
                rec_img_pre = torch.unsqueeze(rec_img_pre, dim=0)
            # calculate the perceptional loss
            loss_pl = loss_fn_vgg(rec_img_after.to(device), rec_img_pre.to(device))     # shape: [1,3,224,224]
            
        # loss_pl = loss_pl.item() 
        loss_pl = loss_pl.squeeze()
        loss = loss_decoder + loss_pl
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_pl=loss_pl)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(loss_decoder=loss_decoder.item())

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if print_memory_usage is True:
        log_max_memory_usage()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}