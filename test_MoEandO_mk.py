import os
import argparse
import subprocess
from tqdm import tqdm
import re
import torch
import time
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils_MoEandO import OfflineMixedTestDataset 
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model_MoEandO_mk import AdaIR


class AdaIRModel(pl.LightningModule):
    def __init__(self, num_experts=5):
        super().__init__()
        self.net = AdaIR(decoder=True, num_experts=num_experts)
    
    def forward(self, x):
        return self.net(x)


def get_tasks_from_ckpt(ckpt_path):
    """从checkpoint文件名解析训练时的任务，推断专家数量"""
    base = os.path.basename(ckpt_path)
    # 匹配文件名中类似 "gsn_sp_jpeg-last.ckpt" 的任务部分
    match = re.search(r"^([a-z_]+)-epoch", base)
    if not match:
        return []
    tasks = match.group(1).split('_')
    valid_tasks = {'gsn', 'sp', 'mb', 'gb', 'jpeg'}
    return [t for t in tasks if t in valid_tasks]


def test_MultiTask(net, dataset, de_types, output_path_base, save_images, log_file=None, ckpt_name=""):
    device = next(net.parameters()).device

    psnr_meters = {de: AverageMeter() for de in de_types}
    ssim_meters = {de: AverageMeter() for de in de_types}

    testloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    for batch in tqdm(testloader, desc="Testing MultiTask"):
        degraded_imgs = batch['LR']
        clean_imgs = batch['HR']
        names = batch['filename']
        de_task_list = batch['de_type']
        degraded_imgs = degraded_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        with torch.no_grad():
            restored = net(degraded_imgs)

        for i in range(len(names)):
            de_type = de_task_list[i]
            if de_type not in de_types:
                continue
            # 计算指标（确保输入是4D张量：[1, C, H, W]）
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored[i:i+1], clean_imgs[i:i+1])
            psnr_meters[de_type].update(temp_psnr, N)
            ssim_meters[de_type].update(temp_ssim, N)

            # 保存恢复图像
            if save_images:
                out_dir = os.path.join(output_path_base, de_type)
                os.makedirs(out_dir, exist_ok=True)
                save_image_tensor(restored[i], os.path.join(out_dir, f"{names[i]}_{de_type}.png"))

    # --- 结果打印与记录 ---
    print(f"\nModel: {os.path.basename(ckpt_name)}")
    
    log_content = []
    log_content.append(f"Model: {os.path.basename(ckpt_name)}")
    # log_content.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}") # 可选：记录时间
    
    for de_type in de_types:
        res_str = f"[{de_type}] PSNR: {psnr_meters[de_type].avg:.4f}, SSIM: {ssim_meters[de_type].avg:.4f}"
        print(res_str)
        log_content.append(res_str)
    
    log_content.append("-" * 40 + "\n")

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        with open(log_file, 'a') as f:
            f.write('\n'.join(log_content))
        print(f"Results appended to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--de_types', nargs='+', default=['gsn', 'gb', 'sp', 'mb', 'jpeg'],
                        help='测试任务列表，示例：--de_types gsn jpeg sp')
    parser.add_argument('--offline_dir', type=str, required=True, help='离线数据目录（含HR和LR子目录）')
    parser.add_argument('--output_path', type=str, default='AdaIR_results/', help='结果保存目录')
    parser.add_argument('--ckpt_name', type=str, required=True, help='模型权重文件路径，如 ./ckpt/adair.ckpt')
    parser.add_argument('--save_images', action='store_true', help='是否保存输出图像')
    parser.add_argument('--log_file', type=str, default=None, help='保存结果的txt文件路径') # 新增
    
    args = parser.parse_args()

    # 设备设置
    device = torch.device(f'cuda:{args.cuda}' if (torch.cuda.is_available() and args.cuda >= 0) else 'cpu')

    # 1. 从checkpoint文件名解析训练时的任务，确定专家数量
    ckpt_tasks = get_tasks_from_ckpt(args.ckpt_name)
    num_experts = len(ckpt_tasks) if ckpt_tasks else len(args.de_types)
    print(f"解析到训练任务: {ckpt_tasks}，专家数量: {num_experts}")

    # 2. 加载模型（使用与训练时一致的专家数量）
    net = AdaIRModel(num_experts=num_experts)
    net = net.load_from_checkpoint(args.ckpt_name, num_experts=num_experts, strict=False)  # 传入专家数量
    net = net.to(device)
    net.eval()

    # 3. 加载测试集
    dataset = OfflineMixedTestDataset(args.offline_dir, de_types=args.de_types)

    # 4. 执行测试
    test_MultiTask(net, dataset, args.de_types, args.output_path, args.save_images, log_file=args.log_file, ckpt_name=args.ckpt_name)