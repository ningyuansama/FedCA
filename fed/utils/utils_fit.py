import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
# def add_fft_perturbation(img_tensor, epsilon=0.01):
#     """
#     img_tensor: [B, C, H, W], torch.float32
#     """
#     img_fft = torch.fft.fft2(img_tensor)
#     img_fft = torch.fft.fftshift(img_fft)
#     noise = epsilon * torch.randn_like(img_fft)
#     img_fft_noised = img_fft + noise
#     img_fft_noised = torch.fft.ifftshift(img_fft_noised)
#     img_noised = torch.fft.ifft2(img_fft_noised).real
#     return img_noised
# #元学习fit
# def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch,
#                   epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler,
#                   save_period, save_dir, local_rank=0, meta_learning=True, fft_epsilon=0.01):
    
#     loss = 0
#     val_loss = 0

#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break

#         images, bboxes = batch
#         if cuda:
#             images = images.cuda(local_rank)
#             bboxes = bboxes.cuda(local_rank)

#         optimizer.zero_grad()

#         # --------------------------#
#         #   生成扰动图像（meta-learning inner-loop）
#         # --------------------------#
#         if meta_learning:
#             images_meta = add_fft_perturbation(images, epsilon=fft_epsilon)
#             outputs_meta = model_train(images_meta)
#             meta_loss = yolo_loss(outputs_meta, bboxes)
#         else:
#             meta_loss = None

#         # --------------------------#
#         #   正常 forward & backward
#         # --------------------------#
#         if not fp16:
#             outputs = model_train(images)
#             loss_value = yolo_loss(outputs, bboxes)

#             if meta_learning:
#                 # 综合 meta_loss 和 normal loss
#                 total_loss = loss_value + meta_loss
#             else:
#                 total_loss = loss_value

#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 outputs = model_train(images)
#                 loss_value = yolo_loss(outputs, bboxes)
#                 if meta_learning:
#                     total_loss = loss_value + meta_loss
#                 else:
#                     total_loss = loss_value

#             scaler.scale(total_loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
#             scaler.step(optimizer)
#             scaler.update()

#         if ema:
#             ema.update(model_train)

#         loss += loss_value.item()

#         if local_rank == 0:
#             pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

#     # --------------------------#
#     #   Validation
#     # --------------------------#
#     model_eval = ema.ema if ema else model_train
#     model_eval.eval()
    
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#         images, bboxes = batch
#         if cuda:
#             images = images.cuda(local_rank)
#             bboxes = bboxes.cuda(local_rank)

#         with torch.no_grad():
#             outputs = model_eval(images)
#             loss_value = yolo_loss(outputs, bboxes)
#         val_loss += loss_value.item()

#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
#         if eval_callback is not None:
#             eval_callback.on_epoch_end(epoch + 1, model_eval)

#         # --------------------------#
#         #   保存模型
#         # --------------------------#
#         save_state_dict = ema.ema.state_dict() if ema else model.state_dict()

#         # 保存最好模型
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

#         # 保存最后一轮
#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

#         print('Epoch: {}/{} | Total Loss: {:.3f} | Val Loss: {:.3f}'.format(
#             epoch + 1, Epoch, loss / epoch_step, val_loss / epoch_step_val
#         ))
        
# fedadg
# def fit_one_epoch(
#     model_train, model, ema, yolo_loss, loss_history, eval_callback,
#     optimizer, epoch, epoch_step, epoch_step_val,
#     gen, gen_val, Epoch, cuda, fp16, scaler,
#     save_period, save_dir, local_rank=0,

#     # ===== FedADG Generator =====
#     generator=None,
#     optimizer_g=None,
#     feat_dim=256,
#     num_classes=4,
#     lambda_align=0.05,
#     device=None
# ):
#     import torch
#     import os
#     import torch.nn.functional as F
#     from tqdm import tqdm
#     from utils.utils import get_lr

#     loss        = 0
#     val_loss    = 0

#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', mininterval=0.3)

#     model_train.train()
#     if generator is not None:
#         generator.train()

#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break

#         images, bboxes = batch
#         if cuda:
#             images = images.to(device)
#             # bboxes 也要放到 device
#             bboxes = [b.to(device) for b in bboxes]

#         optimizer.zero_grad()
#         if optimizer_g is not None:
#             optimizer_g.zero_grad()

#         # ===============================
#         # 1. YOLO Forward
#         # ===============================
#         if not fp16:
#             outputs = model_train(images)
#             loss_det = yolo_loss(outputs, bboxes)
#             loss_value = loss_det

#             # ===============================
#             # 2. FedADG 对齐损失
#             # ===============================
#             if generator is not None:
#                 # 获取特征
#                 feats = model_train(images, return_feat=True)[-1]  # [B, C, H, W]
#                 feats = feats.mean(dim=(2, 3))                     # GAP -> [B, C]

#                 # 安全生成 cls_labels
#                 cls_labels = []
#                 for b in bboxes:
#                     if len(b) > 0:
#                         # b 可能是 [num_boxes, 5] 或 [5]，取第0个box的类别
#                         if b.dim() == 1:
#                             cls_labels.append(int(b[-1].item()))
#                         else:
#                             cls_labels.append(int(b[0, -1].item()))
#                     else:
#                         cls_labels.append(0)

#                 # 对齐 batch size
#                 if len(cls_labels) < feats.size(0):
#                     cls_labels += [0] * (feats.size(0) - len(cls_labels))
#                 elif len(cls_labels) > feats.size(0):
#                     cls_labels = cls_labels[:feats.size(0)]

#                 cls_labels = torch.tensor(cls_labels, device=feats.device)
#                 y_onehot = F.one_hot(cls_labels, num_classes=num_classes).float()

#                 # 生成 fake_feats
#                 z = torch.randn(feats.size(0), feat_dim, device=feats.device)
#                 fake_feats = generator(z, cls_labels)
#                 loss_align = F.mse_loss(feats, fake_feats.detach())
#                 loss_value = loss_value + lambda_align * loss_align

#             # ===============================
#             # 3. Backward（YOLO）
#             # ===============================
#             loss_value.backward()
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), 10.0)
#             optimizer.step()

#             # ===============================
#             # 4. Generator 反向（新增）
#             # ===============================
#             if generator is not None:
#                 z = torch.randn(feats.size(0), feat_dim, device=feats.device)
#                 fake_feats = generator(z, cls_labels)
#                 loss_g = F.mse_loss(fake_feats, feats.detach())
#                 loss_g.backward()
#                 optimizer_g.step()

#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 outputs = model_train(images)
#                 loss_det = yolo_loss(outputs, bboxes)
#                 loss_value = loss_det

#             scaler.scale(loss_value).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), 10.0)
#             scaler.step(optimizer)
#             scaler.update()

#         if ema:
#             ema.update(model_train)

#         loss += loss_value.item()

#         if local_rank == 0:
#             pbar.set_postfix(
#                 loss=loss / (iteration + 1),
#                 lr=get_lr(optimizer)
#             )
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')

#     # ================= Validation =================
#     if ema:
#         model_train_eval = ema.ema
#     else:
#         model_train_eval = model_train.eval()

#     if local_rank == 0:
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', mininterval=0.3)

#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break

#         images, bboxes = batch
#         with torch.no_grad():
#             if cuda:
#                 images = images.to(device)
#                 bboxes = [b.to(device) for b in bboxes]

#             outputs = model_train_eval(images)
#             loss_v = yolo_loss(outputs, bboxes)

#         val_loss += loss_v.item()

#         if local_rank == 0:
#             pbar.set_postfix(val_loss=val_loss / (iteration + 1))
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()

#         loss_history.append_loss(
#             epoch + 1,
#             loss / epoch_step,
#             val_loss / epoch_step_val
#         )

#         if eval_callback is not None:
#             eval_callback.on_epoch_end(epoch + 1, model_train_eval)

#         # 保存 best / last
#         if ema:
#             save_state_dict = ema.ema.state_dict()
#         else:
#             save_state_dict = model.state_dict()

#         if len(loss_history.val_loss) <= 1 or \
#            (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))




# fedavg
# def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
#     loss        = 0
#     val_loss    = 0

#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break

#         images, bboxes = batch
#         with torch.no_grad():
#             if cuda:
#                 images = images.cuda(local_rank)
#                 bboxes = bboxes.cuda(local_rank)
#         #----------------------#
#         #   清零梯度
#         #----------------------#
#         optimizer.zero_grad()
#         if not fp16:
#             #----------------------#
#             #   前向传播
#             #----------------------#
#             # dbox, cls, origin_cls, anchors, strides 
#             outputs = model_train(images)
#             loss_value = yolo_loss(outputs, bboxes)
#             #----------------------#
#             #   反向传播
#             #----------------------#
#             loss_value.backward()
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 #----------------------#
#                 #   前向传播
#                 #----------------------#
#                 outputs         = model_train(images)
#                 loss_value = yolo_loss(outputs, bboxes)

#             #----------------------#
#             #   反向传播
#             #----------------------#
#             scaler.scale(loss_value).backward()
#             scaler.unscale_(optimizer)  # unscale gradients
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
#             scaler.step(optimizer)
#             scaler.update()
#         if ema:
#             ema.update(model_train)

#         loss += loss_value.item()
        
#         if local_rank == 0:
#             pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
#                                 'lr'    : get_lr(optimizer)})
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

#     if ema:
#         model_train_eval = ema.ema
#     else:
#         model_train_eval = model_train.eval()
        
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#         images, bboxes = batch[0], batch[1]
#         with torch.no_grad():
#             if cuda:
#                 images = images.cuda(local_rank)
#                 bboxes = bboxes.cuda(local_rank)
#             #----------------------#
#             #   清零梯度
#             #----------------------#
#             optimizer.zero_grad()
#             #----------------------#
#             #   前向传播
#             #----------------------#
#             outputs     = model_train_eval(images)
#             loss_value  = yolo_loss(outputs, bboxes)

#         val_loss += loss_value.item()
#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
#             pbar.update(1)
            
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
#         # eval_callback.on_epoch_end(epoch + 1, model_train_eval)
#         if eval_callback is not None:
#             eval_callback.on_epoch_end(epoch + 1, model_train_eval)
#         print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
#         print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
#         #-----------------------------------------------#
#         #   保存权值
#         #-----------------------------------------------#
#         if ema:
#             save_state_dict = ema.ema.state_dict()
#         else:
#             save_state_dict = model.state_dict()

#         # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#         #     torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))



import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from utils.utils import get_lr

# def add_fft_perturbation(img_tensor, epsilon=0.01):
#     """ 对图像进行 FFT 干扰，用于模拟 domain shift（元学习任务） """
#     fft = torch.fft.fft2(img_tensor)
#     amplitude = torch.abs(fft)
#     phase = torch.angle(fft)
#     noise = epsilon * torch.randn_like(amplitude)
#     amplitude_noised = amplitude * (1 + noise)
#     fft_noised = amplitude_noised * torch.exp(1j * phase)
#     img_noised = torch.fft.ifft2(fft_noised).real
#     return img_noised


# def fit_one_epoch(
#     model_train, model, ema, yolo_loss, loss_history, eval_callback,
#     optimizer, epoch, epoch_step, epoch_step_val,
#     gen, gen_val, Epoch, cuda, fp16, scaler,
#     save_period, save_dir, local_rank=0,

#     # ===== FedADG Generator =====
#     generator=None,
#     optimizer_g=None,
#     feat_dim=256,
#     num_classes=4,
#     lambda_align=0.05,

#     # ===== 元学习参数 =====
#     meta_learning=True,
#     fft_epsilon=0.01,
#     device=None
# ):
#     loss = 0
#     val_loss = 0

#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{Epoch}', mininterval=0.3)

#     model_train.train()
#     if generator is not None:
#         generator.train()

#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break

#         images, bboxes = batch

#         if cuda:
#             images = images.to(device)
#             bboxes = [b.to(device) for b in bboxes]

#         optimizer.zero_grad()
#         if optimizer_g is not None:
#             optimizer_g.zero_grad()

#         # ===============================
#         # 1. YOLO Forward
#         # ===============================
#         outputs = model_train(images)
#         loss_det = yolo_loss(outputs, bboxes)
#         total_loss = loss_det

#         # ===============================
#         # 2. 元学习: FFT扰动
#         # ===============================
#         if meta_learning:
#             images_meta = add_fft_perturbation(images, fft_epsilon)
#             outputs_meta = model_train(images_meta)
#             meta_loss = yolo_loss(outputs_meta, bboxes)
#             total_loss = total_loss + meta_loss

#         # ===============================
#         # 3. Feature Alignment GAN（按类别循环）
#         # ===============================
#         if generator is not None:
#             feats_list = model_train(images, return_feat=True)
#             # 取最后一层特征
#             feats = feats_list[-1].mean(dim=(2, 3))  # GAP -> [B, C]

#             all_boxes_feats = []
#             for feat_per_img, boxes_per_image in zip(feats, bboxes):
#                 if boxes_per_image.numel() == 0:
#                     continue
#                 if boxes_per_image.dim() == 1:
#                     boxes_per_image = boxes_per_image.unsqueeze(0)
#                 for box_idx in range(boxes_per_image.shape[0]):
#                     cls_box = int(boxes_per_image[box_idx, -1].item())
#                     all_boxes_feats.append((feat_per_img, cls_box))

#             # 按类别计算 GAN 对齐 loss
#             unique_classes = set(cls for _, cls in all_boxes_feats)
#             loss_align = 0.0
#             count = 0
#             for c in unique_classes:
#                 feats_c = torch.stack([f for f, cls in all_boxes_feats if cls == c])
#                 z = torch.randn(feats_c.size(0), feat_dim, device=feats_c.device)
#                 labels_c = torch.full((feats_c.size(0),), c, device=feats_c.device, dtype=torch.long)
#                 fake_feat_c = generator(z, labels_c)
#                 loss_c = F.mse_loss(feats_c, fake_feat_c.detach())
#                 loss_align += loss_c
#                 count += 1
#             if count > 0:
#                 loss_align = loss_align / count
#                 total_loss = total_loss + lambda_align * loss_align

#         # ===============================
#         # 4. Backward YOLO
#         # ===============================
#         if not fp16:
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model_train.parameters(), 10.0)
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 total_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model_train.parameters(), 10.0)
#                 scaler.step(optimizer)
#                 scaler.update()

#         # ===============================
#         # 5. Generator backward（按类别循环）
#         # ===============================
#         if generator is not None and len(all_boxes_feats) > 0:
#             loss_g = 0.0
#             count = 0
#             for c in unique_classes:
#                 feats_c = torch.stack([f for f, cls in all_boxes_feats if cls == c])
#                 z = torch.randn(feats_c.size(0), feat_dim, device=feats_c.device)
#                 labels_c = torch.full((feats_c.size(0),), c, device=feats_c.device, dtype=torch.long)
#                 fake_feat_c = generator(z, labels_c)
#                 loss_c = F.mse_loss(fake_feat_c, feats_c.detach())
#                 loss_g += loss_c
#                 count += 1
#             if count > 0:
#                 loss_g = loss_g / count
#                 loss_g.backward(retain_graph=True)
#                 optimizer_g.step()

#         if ema:
#             ema.update(model_train)

#         loss += loss_det.item()

#         if local_rank == 0:
#             pbar.set_postfix(loss=loss/(iteration+1), lr=get_lr(optimizer))
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')

#     # ===============================
#     # Validation
#     # ===============================
#     model_train_eval = ema.ema if ema else model_train.eval()
#     if local_rank == 0:
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{Epoch}', mininterval=0.3)

#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#         images, bboxes = batch
#         if cuda:
#             images = images.to(device)
#             bboxes = [b.to(device) for b in bboxes]
#         with torch.no_grad():
#             outputs = model_train_eval(images)
#             loss_v = yolo_loss(outputs, bboxes)
#         val_loss += loss_v.item()
#         if local_rank == 0:
#             pbar.set_postfix(val_loss=val_loss/(iteration+1))
#             pbar.update(1)

#     if local_rank == 0:
#         pbar.close()
#         loss_history.append_loss(epoch+1, loss/epoch_step, val_loss/epoch_step_val)
#         if eval_callback:
#             eval_callback.on_epoch_end(epoch+1, model_train_eval)

#         # 保存模型
#         save_state_dict = ema.ema.state_dict() if ema else model.state_dict()
#         if len(loss_history.val_loss) <= 1 or (val_loss/epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


# fedcadg
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from torchvision.ops import roi_align
import torch
import torch.nn as nn

def fit_one_epoch(
    model_train, model, ema, yolo_loss, loss_history, eval_callback,
    optimizer, epoch, epoch_step, epoch_step_val,
    gen, gen_val, Epoch, cuda, fp16, scaler,
    save_period, save_dir, local_rank=0,

    generator=None,
    discriminator=None,
    optimizer_g=None,
    optimizer_d=None,

    feat_dim=256,
    base_align=0.1,
    base_contrast=0,
    base_gan=0.05,
    warmup_epoch=20,

    device=None,
    round=0
):

    loss = 0
    val_loss = 0

    if device is None:
        device = torch.device("cuda" if cuda else "cpu")

    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch+1}/{Epoch}")

    model_train.train()
    if generator: generator.train()
    if discriminator: discriminator.train()

    # ===============================
    # loss schedule
    # ===============================
    progress = min(1.0, round / float(warmup_epoch))
    lambda_align = base_align * progress
    lambda_contrast = base_contrast * progress
    lambda_gan = base_gan * progress

    if local_rank == 0:
        print(f"[Epoch {epoch}] align={lambda_align:.4f}, contrast={lambda_contrast:.4f}, gan={lambda_gan:.4f}")

    loss_stat = {"det": 0.0, "gan": 0.0, "align": 0.0, "contrast": 0.0}

    IMG_SIZE = 448

    # ===============================
    # TRAIN
    # ===============================
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch

        if cuda:
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]

        # ===============================
        # YOLO forward
        # ===============================
        outputs = model_train(images)
        feats_p3, feats_p4, feats_p5 = model_train(images, return_feat=True)
        loss_det = yolo_loss(outputs, bboxes)

        B = images.size(0)

        # ===============================
        # ROI BUILD
        # ===============================
        roi_boxes = []
        roi_labels = []

        new_bboxes = []
        for b in range(len(bboxes)):
            boxes = bboxes[b]
            if isinstance(boxes, list):
                boxes = torch.stack(boxes, dim=0)
            elif boxes.dim() == 1:
                boxes = boxes.view(-1, 6)
            new_bboxes.append(boxes)

        bboxes = new_bboxes
        valid_B = min(B, len(bboxes))

        for b in range(valid_B):
            if len(bboxes[b]) == 0:
                continue

            for box in bboxes[b]:
                if box.numel() < 6:
                    continue

                cls = int(box[1].item())
                x_c, y_c, w, h = box[2:].tolist()

                x_c *= IMG_SIZE
                y_c *= IMG_SIZE
                w *= IMG_SIZE
                h *= IMG_SIZE

                x1 = max(0, x_c - w / 2)
                y1 = max(0, y_c - h / 2)
                x2 = min(IMG_SIZE - 1, x_c + w / 2)
                y2 = min(IMG_SIZE - 1, y_c + h / 2)

                if x2 <= x1 or y2 <= y1:
                    continue

                roi_boxes.append([b, x1, y1, x2, y2])
                roi_labels.append(cls)

        use_roi = len(roi_boxes) > 0

        # ===============================
        # ROI ALIGN (P3 P4 P5)
        # ===============================
        if use_roi:

            roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32, device=device)
            labels_real = torch.tensor(roi_labels, dtype=torch.long, device=device)

            feats_p3_r = roi_align(
                feats_p3, roi_boxes,
                output_size=(14, 14),
                spatial_scale=1/8,
                aligned=True
            )

            feats_p4_r = roi_align(
                feats_p4, roi_boxes,
                output_size=(14, 14),
                spatial_scale=1/16,
                aligned=True
            )

            feats_p5_r = roi_align(
                feats_p5, roi_boxes,
                output_size=(14, 14),
                spatial_scale=1/32,
                aligned=True
            )

        else:
            feats_p3_r = F.adaptive_avg_pool2d(feats_p3, (14, 14))
            feats_p4_r = F.adaptive_avg_pool2d(feats_p4, (14, 14))
            feats_p5_r = F.adaptive_avg_pool2d(feats_p5, (14, 14))

            labels_real = torch.zeros(feats_p5_r.size(0), dtype=torch.long, device=device)

        # ===============================
        # FPN FUSION（后续你接512通道）
        # ===============================
        feats_concat = torch.cat([feats_p3_r, feats_p4_r, feats_p5_r], dim=1)
        feats_real = model_train.fusion_conv(feats_concat)

        # ===============================
        # ===== 1️⃣ 更新判别器 D =====
        # ===============================
        optimizer_d.zero_grad()

        z = torch.randn(feats_real.size(0), feat_dim, device=device)
        feats_fake = generator(z, labels_real)

        real_logits = discriminator(feats_real.detach(), labels_real)
        fake_logits = discriminator(feats_fake.detach(), labels_real)

        loss_d = (
            F.relu(1.0 - real_logits).mean() +
            F.relu(1.0 + fake_logits).mean()
        )

        loss_d.backward()
        optimizer_d.step()

        # ===============================
        # ===== 2️⃣ 更新生成器 G（关键修复）=====
        # ===============================
        optimizer_g.zero_grad()

        z = torch.randn(feats_real.size(0), feat_dim, device=device)
        feats_fake = generator(z, labels_real)

        fake_logits_g = discriminator(feats_fake, labels_real)
        loss_g = -fake_logits_g.mean()

        (lambda_gan * loss_g).backward()
        optimizer_g.step()

        # ===============================
        # ===== 3️⃣ 更新 YOLO（不含 GAN 梯度）=====
        # ===============================
        optimizer.zero_grad()

        # ⚠️ 重新 forward，避免图冲突
        outputs = model_train(images)
        loss_det = yolo_loss(outputs, bboxes)

        feats_p3, feats_p4, feats_p5 = model_train(images, return_feat=True)

        if use_roi:
            feats_p3_r = roi_align(feats_p3, roi_boxes, (14,14), spatial_scale=1/8, aligned=True)
            feats_p4_r = roi_align(feats_p4, roi_boxes, (14,14), spatial_scale=1/16, aligned=True)
            feats_p5_r = roi_align(feats_p5, roi_boxes, (14,14), spatial_scale=1/32, aligned=True)
        else:
            feats_p3_r = F.adaptive_avg_pool2d(feats_p3, (14,14))
            feats_p4_r = F.adaptive_avg_pool2d(feats_p4, (14,14))
            feats_p5_r = F.adaptive_avg_pool2d(feats_p5, (14,14))

        # ⭐ concat
        feats_concat = torch.cat([feats_p3_r, feats_p4_r, feats_p5_r], dim=1)

        # ⭐ fusion
        feats_real = model_train.fusion_conv(feats_concat)

        # ===== Align =====
        z = torch.randn(feats_real.size(0), feat_dim, device=device)
        feats_fake = generator(z, labels_real).detach()  # ❗断开G

        align_loss = F.l1_loss(feats_fake, feats_real)

        # ===== Contrast =====
        f_real = F.normalize(feats_real.flatten(1), dim=1)
        f_fake = F.normalize(feats_fake.flatten(1), dim=1)

        feats_all = torch.cat([f_real, f_fake], dim=0)
        labels_all = torch.cat([labels_real, labels_real], dim=0)

        sim = torch.matmul(feats_all, feats_all.T)

        mask = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1)).float()

        exp_sim = torch.exp(sim / 0.07)
        exp_sim = exp_sim * (1 - torch.eye(exp_sim.size(0), device=device))

        pos = (exp_sim * mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) + 1e-8

        loss_contrast = -torch.log(pos / all_sum).mean()

        total_loss = (
            loss_det
            + lambda_align * align_loss
            + lambda_contrast * loss_contrast
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 10.0)
        optimizer.step()

        if ema:
            ema.update(model_train)

        loss += loss_det.item()

        loss_stat["det"] += loss_det.item()
        loss_stat["gan"] += (loss_d.item() + loss_g.item())
        loss_stat["align"] += align_loss.item()
        loss_stat["contrast"] += loss_contrast.item()

        if local_rank == 0:
            pbar.set_postfix({
                "det": loss_stat["det"]/(iteration+1),
                "align": loss_stat["align"]/(iteration+1),
                "contrast": loss_stat["contrast"]/(iteration+1),
            })
            pbar.update(1)

    # ===============================
    # Validation
    # ===============================
    model_train_eval = ema.ema if ema else model_train.eval()
    if local_rank == 0:
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{Epoch}', mininterval=0.3)

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch
        if cuda:
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]
        with torch.no_grad():
            outputs = model_train_eval(images)
            loss_v = yolo_loss(outputs, bboxes)
        val_loss += loss_v.item()
        if local_rank == 0:
            pbar.set_postfix(val_loss=val_loss/(iteration+1))
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch+1, loss/epoch_step, val_loss/epoch_step_val)
        if eval_callback:
            eval_callback.on_epoch_end(epoch+1, model_train_eval)

        # 保存模型
        save_state_dict = ema.ema.state_dict() if ema else model.state_dict()
        if len(loss_history.val_loss) <= 1 or (val_loss/epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))