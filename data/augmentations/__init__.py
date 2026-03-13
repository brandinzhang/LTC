# data/augmentations/__init__.py
from torchvision import transforms
import torch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def get_transform(transform_type='imagenet', image_size=224, args=None):
    if transform_type == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])
        return (train_transform, test_transform)

    elif transform_type == 'clip':
        # Use CLIP statistics; RandomResizedCrop is more suitable for contrastive learning during training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(CLIP_MEAN), std=torch.tensor(CLIP_STD)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(CLIP_MEAN), std=torch.tensor(CLIP_STD)),
        ])
        return (train_transform, test_transform)

    else:
        raise NotImplementedError
