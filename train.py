import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset_utils import OfflineMixedTrainDataset
from net.model import OMoE-Net
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from utils.loss_utils import OrthogonalLoss, FrequencyLoss, CharbonnierLoss

class UniqueCheckpoint(ModelCheckpoint):
    def __init__(self, unique_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique_id = unique_id

    @property
    def state_key(self):
        return f"ModelCheckpoint_{self.unique_id}"

class LossLoggerCallback(Callback):
    def __init__(self, 
                 global_filename="all_tasks_loss_history.txt", 
                 current_filename="current_stage_loss.txt", 
                 log_every_n_steps=50, 
                 task_suffix=""):
        super().__init__()
        self.global_filename = global_filename
        self.current_filename = current_filename
        self.log_every_n_steps = log_every_n_steps
        self.task_suffix = task_suffix
        
        self.save_dir = None
        self.global_path = None
        self.current_path = None

    def _setup_paths(self, trainer):
        if self.save_dir is None:
            try:
                if hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                    self.save_dir = trainer.logger.save_dir
                elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'dir'):
                    self.save_dir = trainer.logger.experiment.dir
                else:
                    self.save_dir = trainer.default_root_dir
            except:
                self.save_dir = "." 
        
        self.global_path = os.path.join(self.save_dir, self.global_filename)
        self.current_path = os.path.join(self.save_dir, self.current_filename)

    def on_train_start(self, trainer, pl_module):
        self._setup_paths(trainer)
        os.makedirs(os.path.dirname(self.current_path), exist_ok=True)
        with open(self.current_path, "w") as f:
            f.write("") 
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            if self.current_path is None:
                self._setup_paths(trainer)
            
            metrics = trainer.callback_metrics
            current_step = trainer.global_step
            
            log_parts = [
                f"Task: {self.task_suffix}",
                f"Step: {current_step}", 
                f"Epoch: {trainer.current_epoch}"
            ]
            
            if "gen_total_loss" in metrics:
                log_parts.append(f"Total: {metrics['gen_total_loss'].item():.6f}")
            if "gen_rec_loss" in metrics:
                log_parts.append(f"Rec: {metrics['gen_rec_loss'].item():.6f}")
            if "gen_fft_loss" in metrics:
                log_parts.append(f"FFT: {metrics['gen_fft_loss'].item():.6f}")
            if "ortho_loss" in metrics:
                log_parts.append(f"Ortho: {metrics['ortho_loss'].item():.6f}")
            if "guide_loss" in metrics:
                log_parts.append(f"Guide: {metrics['guide_loss'].item():.6f}")
            
            for key, value in metrics.items():
                if key.startswith("rec_loss/task_"):
                    short_key = key.replace("rec_loss/", "") 
                    log_parts.append(f"{short_key}: {value.item():.6f}")

            log_str = " | ".join(log_parts) + "\n"

            with open(self.global_path, "a") as f:
                f.write(log_str)
            
            with open(self.current_path, "a") as f:
                f.write(log_str)

    def on_train_end(self, trainer, pl_module):
        if self.current_path and os.path.exists(self.current_path):
            img_name = f"loss_visualization_{self.task_suffix}.png"
            output_img_path = os.path.join(os.path.dirname(self.current_path), img_name)

            self.draw_loss_curves(self.current_path, output_img_path)

    def draw_loss_curves(self, log_path, output_path):
        if not os.path.exists(log_path):
            return
        data = {} 
        steps = []

        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(" | ")
            row_data = {}
            for part in parts:
                if ":" in part:
                    key, val = part.split(":")
                    key = key.strip()
                    try:
                        if key == "Task": continue
                        val = float(val.strip())
                        row_data[key] = val
                    except: pass
            
            if "Step" in row_data:
                step = row_data["Step"]
                for k, v in row_data.items():
                    if k not in ["Step", "Epoch"]:
                        if k not in data: data[k] = []
                        data[k].append((step, v)) 

        if not data: return

        plt.switch_backend('agg') 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        main_keys = ["Total", "Rec", "Ortho"]
        has_main = False
        for key in main_keys:
            if key in data:
                xy = data[key]
                xs = [p[0] for p in xy]
                ys = [p[1] for p in xy]
                ax1.plot(xs, ys, label=key, linewidth=2)
                has_main = True
        
        ax1.set_title(f"Global Training Losses ({self.task_suffix})") 
        ax1.set_ylabel("Loss Value")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        if has_main: ax1.set_yscale('log')

        has_task = False
        for key, xy_list in data.items():
            if key not in main_keys and key not in ["Step", "Epoch"]:
                xs = [p[0] for p in xy_list]
                ys = [p[1] for p in xy_list]
                ax2.plot(xs, ys, label=key, alpha=0.7)
                has_task = True
        
        ax2.set_title("Per-Task Reconstruction Losses")
        ax2.set_xlabel("Global Steps")
        ax2.set_ylabel("Loss Value")
        ax2.grid(True, which="both", alpha=0.3)
        if has_task: ax2.legend(loc='upper right', ncol=2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

class OMoE-NetModel(pl.LightningModule):
    def __init__(self, pretrained_ckpt=None, freeze_encoder=False):
        super().__init__()
        self.automatic_optimization = False
        self.loss_fn = CharbonnierLoss(eps=1e-3)
        self.freq_loss_fn = FrequencyLoss(loss_weight=0.1)
        self.task2id = {'gsn': 0, 'sp': 1, 'jpeg': 2, 'gb': 3, 'mb': 4}  
        self.current_tasks = opt.de_type if isinstance(opt.de_type, list) else [opt.de_type]
        self.old_tasks = self._extract_old_tasks_from_ckpt(pretrained_ckpt) if pretrained_ckpt else []
        self.all_tasks = list(dict.fromkeys(self.current_tasks + self.old_tasks))
        self.task_id_map = {task: self.task2id[task] for task in self.all_tasks}
        self.num_experts = len(self.all_tasks)
        self.net = OMoE-Net(decoder=True, num_experts=self.num_experts)
        self.orthogonal_loss = OrthogonalLoss(eps=opt.orthogonal_eps)
        self.orthogonal_loss_weight = opt.orthogonal_loss_weight
        self.guidance_loss_weight = getattr(opt, 'guidance_loss_weight', 0.1)
        
        if freeze_encoder:
            self.freeze_encoder_layers(freeze_ratio=opt.freeze_ratio)
        
        if pretrained_ckpt is not None:
            self._load_pretrained_and_freeze(pretrained_ckpt)
            
    def _extract_old_tasks_from_ckpt(self, ckpt_path):
        import os
        if not ckpt_path: return []
        filename = os.path.basename(ckpt_path)
        name_no_ext = os.path.splitext(filename)[0].replace("-last", "")
        potential_tasks = name_no_ext.split('_')
        valid_tasks_keys = set(self.task2id.keys())
        return [t for t in potential_tasks if t in valid_tasks_keys]

    def _load_pretrained_and_freeze(self, ckpt_path):
        print(f"Loading pretrained: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        self.net.load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id, task_id], degrad_patch, clean_patch) = batch
        if isinstance(de_id, list):
            domain_labels = [self.task_id_map[task] for task in de_id]
        else:
            domain_labels = self.task_id_map[de_id]
        domain_labels = torch.tensor(domain_labels, dtype=torch.long, device=self.device)

        optimizer = self.optimizers()

        self.toggle_optimizer(optimizer)
        restored, encoder_feat, expert_feats = self.net(degrad_patch, return_feat=True)
        pixel_loss = self.loss_fn(restored, clean_patch)
        fft_loss = self.freq_loss_fn(restored.float(), clean_patch.float())
        rec_loss = pixel_loss + fft_loss

        ortho_loss = self.orthogonal_loss(expert_feats)
        
        guide_loss = torch.tensor(0.0, device=self.device)
        
        guidance_loss_sum = 0.0
        guidance_count = 0

        for i in range(len(domain_labels)):
            task = domain_labels[i].item()
            task_rec_loss = self.loss_fn(restored[i], clean_patch[i])
            self.log(f"rec_loss/task_{task}", task_rec_loss, sync_dist=True)

        if expert_feats.shape[1] >= len(self.task_id_map):  
            gaussian_noise = degrad_patch - clean_patch  
            salt_mask = (degrad_patch == 1.0).float()
            pepper_mask = (degrad_patch == 0.0).float()
            salt_pepper_mask = salt_mask + pepper_mask  

            jpeg_block_mask = torch.zeros_like(degrad_patch)
            if degrad_patch.shape[2] >= 8 and degrad_patch.shape[3] >= 8:
                for i in range(8, degrad_patch.shape[2], 8):
                    jpeg_block_mask[:, :, i-1:i, :] = 1.0 
                for j in range(8, degrad_patch.shape[3], 8):
                    jpeg_block_mask[:, :, :, j-1:j] = 1.0 

            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                            device=self.device, dtype=degrad_patch.dtype)
            laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3) 
            gb_high_freq = torch.abs(torch.nn.functional.conv2d(
                degrad_patch,
                laplacian_kernel.repeat(degrad_patch.size(1), 1, 1, 1), 
                padding=1, groups=degrad_patch.size(1) 
            ))

            sobel_x_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                          device=self.device, dtype=degrad_patch.dtype)
            sobel_y_kernel = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                          device=self.device, dtype=degrad_patch.dtype)

            sobel_x = torch.nn.functional.conv2d(
                degrad_patch,
                sobel_x_kernel.repeat(degrad_patch.size(1), 1, 1, 1), 
                padding=1, groups=degrad_patch.size(1)
            )
            sobel_y = torch.nn.functional.conv2d(
                degrad_patch,
                sobel_y_kernel.repeat(degrad_patch.size(1), 1, 1, 1), 
                padding=1, groups=degrad_patch.size(1)
            )
            mb_dir_mask = torch.atan2(sobel_y, sobel_x) 

            unique_tasks = torch.unique(domain_labels)
            for task_id in unique_tasks:
                mask = (domain_labels == task_id)
                if not torch.any(mask): continue

                if task_id == 0: 
                    expert_idx = 0; feat = expert_feats[:, expert_idx][mask]; degenerate = gaussian_noise[mask].flatten(1)
                elif task_id == 1: 
                    expert_idx = 1; feat = expert_feats[:, expert_idx][mask]; degenerate = salt_pepper_mask[mask].flatten(1)
                elif task_id == 2: 
                    expert_idx = 2; feat = expert_feats[:, expert_idx][mask]; degenerate = jpeg_block_mask[mask].flatten(1)
                elif task_id == 3: 
                    expert_idx = 3; feat = expert_feats[:, expert_idx][mask]; degenerate = gb_high_freq[mask].flatten(1)
                elif task_id == 4: 
                    expert_idx = 4; feat = expert_feats[:, expert_idx][mask]; degenerate = mb_dir_mask[mask].flatten(1)
                else:
                    continue 

                feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat_flat = feat_pooled.flatten(1)

                proj_dim = 512
                proj_key = f"proj_matrix_{task_id}"
                if not hasattr(self, proj_key):
                    setattr(self, proj_key, torch.randn(feat_flat.shape[1], proj_dim, device=self.device))
                
                feat_flat = torch.matmul(feat_flat, getattr(self, proj_key))
                
                if degenerate.shape[1] > proj_dim:
                    if not hasattr(self, f"degen_proj_{task_id}"):
                        setattr(self, f"degen_proj_{task_id}", torch.randn(degenerate.shape[1], proj_dim, device=self.device))
                    degenerate = torch.matmul(degenerate, getattr(self, f"degen_proj_{task_id}"))
                else:
                    degenerate = torch.nn.functional.pad(degenerate, (0, proj_dim - degenerate.shape[1]))

                feat_norm = torch.nn.functional.normalize(feat_flat, dim=1)
                degenerate_norm = torch.nn.functional.normalize(degenerate, dim=1)
                
                corr = torch.mean(torch.sum(feat_norm * degenerate_norm, dim=1))
                self.log(f"corr/expert{expert_idx}_task{task_id}", corr.item(), sync_dist=True, prog_bar=True)
                
                current_guide_loss = 1.0 - corr
                guidance_loss_sum += current_guide_loss
                guidance_count += 1
                
        guide_loss = guidance_loss_sum / guidance_count if guidance_count > 0 else torch.tensor(0.0, device=self.device)
        
        cur_guide_weight = self.guidance_loss_weight if self.current_epoch < 50 else self.guidance_loss_weight * 0.1

        gen_total_loss = rec_loss + \
                         (self.orthogonal_loss_weight * ortho_loss) + \
                         (cur_guide_weight * guide_loss)

        self.log("gen_rec_loss", rec_loss, sync_dist=True)
        self.log("ortho_loss", ortho_loss, sync_dist=True)
        self.log("guide_loss", guide_loss, sync_dist=True)
        self.log("gen_fft_loss", fft_loss, sync_dist=True)
        self.log("gen_total_loss", gen_total_loss, sync_dist=True, on_step=True, on_epoch=True)

        optimizer.zero_grad()
        self.manual_backward(gen_total_loss)
        
        if hasattr(opt, 'grad_clip_val') and opt.grad_clip_val > 0:
            self.clip_gradients(optimizer, gradient_clip_val=opt.grad_clip_val, gradient_clip_algorithm="norm")
        
        optimizer.step()
        self.untoggle_optimizer(optimizer)
        
        return gen_total_loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_optimizers(self):
        gen_params = filter(lambda p: p.requires_grad, self.net.parameters())
        gen_optimizer = optim.AdamW(gen_params, lr=opt.lr)
        gen_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=gen_optimizer,
            warmup_epochs=15,
            max_epochs=opt.epochs
        )
        return [gen_optimizer], [gen_scheduler]
    
    def freeze_encoder_layers(self, freeze_ratio=1.0):
        print(f"Freeze rate = {freeze_ratio}")
        encoder_layers = ['encoder_level1', 'encoder_level2', 'encoder_level3']
        num_to_freeze = int(len(encoder_layers) * freeze_ratio)
        for name, module in self.net.named_children():
            if name in encoder_layers[:num_to_freeze]:
                print(f"Freezing {name}")
                for param in module.parameters():
                    param.requires_grad = False


def main():
    print("Options")
    print(opt)
    
    torch.set_float32_matmul_precision('high')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="OMoE-Net-Train", offline=True)
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = OfflineMixedTrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.num_workers,
        drop_last=True,
        prefetch_factor=4
    )
    
    current_task = opt.de_type if isinstance(opt.de_type, str) else opt.de_type[0]
    
    if opt.pretrained_ckpt:
        basename = os.path.basename(opt.pretrained_ckpt)
        prev_name = basename.replace("-last.ckpt", "").replace(".ckpt", "")
        experiment_name = f"{prev_name}_{current_task}"
    else:
        experiment_name = current_task

    checkpoint_last = UniqueCheckpoint(
        unique_id="last",
        dirpath=opt.ckpt_dir,
        filename=f"{experiment_name}-last",
        every_n_epochs=1,
        save_top_k=1,
        monitor=None,
        save_last=False
    )
    
    checkpoint_history = UniqueCheckpoint(
        unique_id="history",
        dirpath=opt.ckpt_dir,
        filename=f"{experiment_name}-{{epoch:02d}}",
        every_n_epochs=1,
        save_top_k=-1,
        monitor=None,
        save_last=False
    )
    
    if isinstance(opt.de_type, list):
        task_suffix = "_".join(opt.de_type)
    else:
        task_suffix = str(opt.de_type)

    loss_logger = LossLoggerCallback(
        global_filename="all_tasks_loss_history.txt",
        current_filename="current_stage_loss.txt",
        log_every_n_steps=50,
        task_suffix=task_suffix  
    )

    model = OMoE-NetModel(pretrained_ckpt=opt.pretrained_ckpt, freeze_encoder=opt.freeze_encoder)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_last, checkpoint_history, loss_logger],
        precision="16-mixed",
    )

    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()
