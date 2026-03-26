import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
    

class OfflineMixedTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gt_dir = os.path.join(args.offline_dir, 'HR')
        self.lr_dir = os.path.join(args.offline_dir, 'LR')
        self.patch_size = args.patch_size
        self.toTensor = ToTensor()

        self.task2id = {
            'gsn': 0, 'sp': 1, 'jpeg': 2, 
            'gb': 3, 'mb': 4
        }
        self.degradation_suffixes = {
            'gsn': '_gsn.png', 'sp': '_sp.png', 
            'mb': '_mb.png', 'gb': '_gb.png', 'jpeg': '_jpeg.png'
        }

        self.new_tasks = args.de_type if isinstance(args.de_type, list) else [args.de_type]
        self.new_tasks = [t for t in self.new_tasks if t in self.degradation_suffixes]
        assert self.new_tasks, "传入的--de_type中没有有效的任务类型！"

        self.old_tasks = []
        if args.pretrained_ckpt:
            self.old_tasks = self._extract_old_tasks_from_ckpt(args.pretrained_ckpt)
            self.old_tasks = [t for t in self.old_tasks if t not in self.new_tasks]

        self.all_tasks = self.new_tasks + self.old_tasks
        print(f"新任务: {self.new_tasks} | 旧任务: {self.old_tasks if self.old_tasks else '无'}")


        self.task_probs = self._compute_sampling_probs()
        print(f"任务采样比例: {dict(zip(self.all_tasks, self.task_probs))}")


        self.image_names = self._load_valid_image_names()
        self._check_degraded_files()

    def _extract_old_tasks_from_ckpt(self, ckpt_path):

        import os
        
        if not ckpt_path:
            return []

        filename = os.path.basename(ckpt_path)
        
        name_no_ext = os.path.splitext(filename)[0]
        
        if name_no_ext.endswith("-last"):
            raw_task_str = name_no_ext[:-5]
        else:
            raw_task_str = name_no_ext.split("-")[0]

        potential_tasks = raw_task_str.split('_')
        
        valid_extracted_tasks = [t for t in potential_tasks if t in self.degradation_suffixes]
        
        return valid_extracted_tasks

    def _compute_sampling_probs(self):
        num_new = len(self.new_tasks)
        num_old = len(self.old_tasks)

        if num_old == 0:
            return [1.0 / num_new for _ in self.new_tasks]
        elif num_new == 1 and num_old == 0:
            return [1.0]
        else:
            new_prob_per_task = 0.5 / num_new
            old_prob_per_task = 0.5 / num_old
            return [new_prob_per_task for _ in self.new_tasks] + [old_prob_per_task for _ in self.old_tasks]

    def _load_valid_image_names(self):
        image_names = []
        for f in sorted(os.listdir(self.gt_dir)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            name_no_ext = os.path.splitext(f)[0]
            has_valid_degraded = False
            for task in self.all_tasks:
                degraded_name = name_no_ext + self.degradation_suffixes[task]
                if os.path.exists(os.path.join(self.lr_dir, degraded_name)):
                    has_valid_degraded = True
                    break
            if has_valid_degraded:
                image_names.append(f)
        assert image_names, f"在{self.gt_dir}中未找到有效的HR图像（或对应退化图不存在）"
        return image_names

    def _check_degraded_files(self):
        sample_names = self.image_names[:5] if len(self.image_names) > 5 else self.image_names
        for base_name in sample_names:
            name_no_ext = os.path.splitext(base_name)[0]
            for task in self.all_tasks:
                degraded_path = os.path.join(self.lr_dir, name_no_ext + self.degradation_suffixes[task])
                if not os.path.exists(degraded_path):
                    raise FileNotFoundError(f"任务[{task}]的退化图像不存在: {degraded_path}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_name = self.image_names[idx]
        name_no_ext = os.path.splitext(base_name)[0]
        clean_path = os.path.join(self.gt_dir, base_name)

        selected_task = random.choices(self.all_tasks, weights=self.task_probs, k=1)[0]

        suffix = self.degradation_suffixes[selected_task]
        degraded_path = os.path.join(self.lr_dir, name_no_ext + suffix)
        clean_img = Image.open(clean_path).convert('RGB')
        degraded_img = Image.open(degraded_path).convert('RGB')

        assert clean_img.size == degraded_img.size, \
            f"尺寸不匹配: {clean_path} ({clean_img.size}) vs {degraded_path} ({degraded_img.size})"

        w, h = clean_img.size
        ps = self.patch_size
        if w < ps or h < ps:
            clean_img = clean_img.resize((ps, ps), Image.BILINEAR)
            degraded_img = degraded_img.resize((ps, ps), Image.BILINEAR)
            left, top = 0, 0
        else:
            left = random.randint(0, w - ps)
            top = random.randint(0, h - ps)
        clean_crop = clean_img.crop((left, top, left + ps, top + ps))
        degraded_crop = degraded_img.crop((left, top, left + ps, top + ps))

        clean_np, degraded_np = random_augmentation(np.array(clean_crop), np.array(degraded_crop))
        clean_tensor = self.toTensor(clean_np)
        degraded_tensor = self.toTensor(degraded_np)

        task_id = self.task2id[selected_task]

        return {
            'LR': degraded_tensor,
            'HR': clean_tensor,
            'task_id': task_id,
            'filename': name_no_ext,
            'de_type': selected_task
        }

    
class OfflineMixedTestDataset(Dataset):
    def __init__(self, offline_dir, de_types=None):
        self.root = offline_dir  # e.g., "data/Test"
        self.hr_dir = os.path.join(self.root, "HR")

        self.supported_de_types = ['gsn', 'sp', 'gb', 'mb', 'jpeg']

        if de_types is None:
            self.de_types = self.supported_de_types
        else:
            self.de_types = [de for de in de_types if de in self.supported_de_types]
        
        self.toTensor = ToTensor()

        self.filenames = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.filenames) * len(self.de_types)

    def __getitem__(self, index):
        img_idx = index // len(self.de_types)
        de_idx = index % len(self.de_types)

        filename = self.filenames[img_idx]
        de_type = self.de_types[de_idx]

        hr_path = os.path.join(self.hr_dir, filename)

        lr_dir = os.path.join(self.root, f"LR_{de_type}")
        lr_path = os.path.join(lr_dir, filename)

        hr_img = np.array(Image.open(hr_path).convert("RGB"))
        lr_img = np.array(Image.open(lr_path).convert("RGB"))

        hr_img = crop_img(hr_img, base=16)
        lr_img = crop_img(lr_img, base=16)

        hr_tensor = self.toTensor(hr_img)
        lr_tensor = self.toTensor(lr_img)

        return {
            'LR': lr_tensor,
            'HR': hr_tensor,
            'filename': filename,
            'de_type': de_type,
        }
