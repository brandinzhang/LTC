# data/cifar100.py
from torchvision.datasets import CIFAR100
from copy import deepcopy
from typing import Any, Tuple
from PIL import Image
import numpy as np

class CIFAR100_Base(CIFAR100):
    def __init__(self, root=None, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=(split != 'test'), transform=transform,
                         target_transform=target_transform, download=download)
        self.data = np.array(self.data)          # (N, 32, 32, 3)
        self.targets = np.array(self.targets)    # (N,)
        self.uq_idxs = np.arange(len(self))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int, int]:
        img = Image.fromarray(self.data[idx])
        target: Any = int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        uq_idx = int(self.uq_idxs[idx])
        return img, target, uq_idx, uq_idx


class CIFAR100Dataset(CIFAR100_Base):
    def __init__(self, root=None, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root=root, split=split, transform=transform,
                         target_transform=target_transform, download=download)


def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    np.random.seed(0)
    m = int(prop_indices_to_subsample * len(dataset))
    subsample_indices = np.random.choice(np.arange(len(dataset)), size=(m,), replace=False)
    return subsample_indices

def subsample_dataset(dataset, idxs):
    if len(idxs) > 0:
        idxs = np.asarray(idxs)
        dataset.data = dataset.data[idxs]
        dataset.targets = np.asarray(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset
    else:
        return None

def subsample_classes(dataset, include_classes=tuple(range(80))):
    cls_idxs = [i for i, t in enumerate(dataset.targets) if t in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)
    train_idxs, val_idxs = [], []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=int(val_split * len(cls_idxs)))
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_); val_idxs.extend(v_)
    return train_idxs, val_idxs

def get_cifar100_datasets(train_transform,
                          test_transform,
                          train_classes=tuple(range(80)),
                          prop_train_labels=0.8,
                          split_train_val=False,
                          seed=0,
                          data_root=None):

    np.random.seed(seed)
    whole_training_set = CIFAR100Dataset(root=data_root, split='train', transform=train_transform, download=True)

    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    if val_dataset_labelled_split is not None:
        val_dataset_labelled_split.transform = test_transform

    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    whole_test_dataset = CIFAR100Dataset(root=data_root, split='test', transform=test_transform, download=True)
    test_dataset = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)

    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    return train_dataset_labelled, test_dataset, train_dataset_unlabelled
