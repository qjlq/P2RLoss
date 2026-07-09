# -*- coding: utf-8 -*-

import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter, Image

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        # 確保 x 是 PIL Image，因為 ImageFilter 屬於 PIL
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class NormalSample(object):
    def __init__(self, mean, std, crop_size=(256, 256), resize_factor=0.3, train=False):
        self.half_h, self.half_w = crop_size
        self.train = train
        self.scale_factor = resize_factor

        self.im2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # 強增強管道（若需要對 Unlabeled Tensor 使用，需注意與 PIL 轉換）
        self.strong_aug = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.25),
            self.im2tensor
        ])

    def process_lable(self, image, dotseq):
        """ image 預期為 Tensor (3, H, W)，dotseq 為 Tensor (N, 2) """
        if self.train:
            images, dotseqs = self.crop_and_resize(image, dotseq)
        else:
            images, dotseqs = image.unsqueeze(0), [dotseq]
            ih, iw = images.shape[-2:]
            if ih != self.half_h or iw != self.half_w:
                rh, rw = self.half_h / ih, self.half_w / iw
                images = F.interpolate(
                    images, (self.half_h, self.half_w),
                    mode='bilinear', align_corners=False
                )
                for j, dots in enumerate(dotseqs):
                    if dots.size(0) > 0:
                        scaled = dots.clone().float()
                        scaled[:, 0] *= rw  # x
                        scaled[:, 1] *= rh  # y
                        dotseqs[j] = scaled

        # 32倍數對齊
        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            h, w = images.shape[-2:]

        # 水平翻轉數據增強
        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
                if dotseqs[i].numel() > 0:
                    dotseqs[i][:, 0] = w - dotseqs[i][:, 0] - 1

        return images, dotseqs

    def process_unlabel(self, image):
        """ 建議：在半監督訓練中激活強增強（此處視您的架構決定是否調用） """
        if self.train:
            images = self.crop_and_resize(image)
        else:
            raise NotImplementedError("Should not happen...")

        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            h, w = images.shape[-2:]

        for i in range(images.size(0)):
            if self.train and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
                
        return images

    def crop_and_resize(self, image, dotseq=None, num_patches=1):
        imh, imw = image.shape[-2:]

        scale = random.random() * (self.scale_factor * 2) + (1 - self.scale_factor)
        crop_h = int(self.half_h / scale + 0.5)
        crop_w = int(self.half_w / scale + 0.5)

        if crop_h > imh or crop_w > imw:
            padw = max(0, crop_w - imw)
            padh = max(0, crop_h - imh)
            image = F.pad(image, (0, padw, 0, padh), mode='constant', value=0)
            imh, imw = image.shape[-2:]

        crop_imgs, crop_dots = [], []
        rh, rw = self.half_h / crop_h, self.half_w / crop_w

        for _ in range(num_patches):
            start_h = random.randint(0, imh - crop_h)
            start_w = random.randint(0, imw - crop_w)
            end_h = start_h + crop_h
            end_w = start_w + crop_w

            crop_img = image[:, start_h:end_h, start_w:end_w]
            crop_imgs.append(crop_img)

            if dotseq is not None:
                idx = (
                    (dotseq[:, 0] >= start_w) & (dotseq[:, 0] <= end_w) &
                    (dotseq[:, 1] >= start_h) & (dotseq[:, 1] <= end_h)
                )
                selected_dot = dotseq[idx].clone()
                if selected_dot.size(0) > 0:
                    selected_dot[:, 0] = (selected_dot[:, 0] - start_w) * rw  # x
                    selected_dot[:, 1] = (selected_dot[:, 1] - start_h) * rh  # y
                crop_dots.append(selected_dot)

        crop_imgs = torch.stack(crop_imgs, dim=0)
        crop_imgs = F.interpolate(
            crop_imgs, (self.half_h, self.half_w),
            mode='bilinear', align_corners=False
        )

        if dotseq is not None:
            return crop_imgs, crop_dots
        else:
            return crop_imgs

def jpg2id(jpg):
    return jpg.replace('.jpg', '')