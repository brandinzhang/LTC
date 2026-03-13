import os
import numpy as np
from copy import deepcopy
from torchvision.datasets import ImageFolder


class ImageNet100Merged(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.uq_idxs = np.arange(len(self), dtype=np.int64)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        uq_idx = int(self.uq_idxs[index])
        return img, target, uq_idx, index


def _subset_by_indices(ds, idxs):
    ds = deepcopy(ds)
    idxs = np.asarray(idxs)
    ds.samples = [ds.samples[i] for i in idxs]
    ds.targets = [ds.targets[i] for i in idxs]
    ds.uq_idxs = ds.uq_idxs[idxs]
    return ds


def _remap_targets(ds, old_to_new):
    ds = deepcopy(ds)
    ds.samples = [(p, old_to_new[t]) for (p, t) in ds.samples]
    ds.targets = [old_to_new[t] for t in ds.targets]
    if hasattr(ds, "class_to_idx"):
        ds.class_to_idx = {k: old_to_new[v] for k, v in ds.class_to_idx.items() if v in old_to_new}
    return ds


def _per_class_subsample_indices(targets, include_classes, prop, seed):
    """Subsample a proportion 'prop' per class from include_classes, keeping at least 1 image."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    keep = []
    for c in include_classes:
        cls_idx = np.where(targets == c)[0]
        if len(cls_idx) == 0:
            continue
        k = max(1, int(np.floor(len(cls_idx) * prop)))
        sel = rng.choice(cls_idx, size=k, replace=False)
        keep.extend(sel.tolist())
    keep = np.array(sorted(keep))
    return keep


def get_imagenet100_merged_datasets(
    train_transform,
    test_transform,
    data_root,
    known_count=50,
    known_class_list=None,
    prop_train_labels=0.5,
    seed=0
):
    train_set_raw = ImageNet100Merged(root=os.path.join(data_root, "train"), transform=train_transform)
    val_set_raw   = ImageNet100Merged(root=os.path.join(data_root, "val"),   transform=test_transform)

    num_classes = len(train_set_raw.classes)
    assert num_classes == 100, f"Expect 100 classes, got {num_classes}"

    if known_class_list is None:
        known_old = list(range(known_count))  # 0..known_count-1
    else:
        if isinstance(known_class_list[0], str):
            name_to_old = train_set_raw.class_to_idx  # {'n01440764': 0, ...}
            known_old = [name_to_old[nm] for nm in known_class_list]
        else:
            known_old = list(known_class_list)

    unknown_old = [i for i in range(num_classes) if i not in known_old]

    remap_order = known_old + unknown_old
    old_to_new = {old: new for new, old in enumerate(remap_order)}

    whole_train = _remap_targets(train_set_raw, old_to_new)
    val_all     = _remap_targets(val_set_raw,   old_to_new)

    known_new = list(range(len(known_old)))  # 0..known_count-1
    keep_labelled = _per_class_subsample_indices(
        targets=whole_train.targets,
        include_classes=known_new,
        prop=prop_train_labels,
        seed=seed
    )
    train_labelled = _subset_by_indices(whole_train, keep_labelled)

    all_idx = np.arange(len(whole_train))
    keep_unlabelled = np.setdiff1d(all_idx, keep_labelled)
    train_unlabelled = _subset_by_indices(whole_train, keep_unlabelled)

    meta = {
        "known_new": known_new,
        "unknown_new": list(range(len(known_old), num_classes)),
        "class_order_old_to_new": old_to_new,
        "class_names_in_order": [train_set_raw.classes[o] for o in remap_order],
    }
    return train_labelled, val_all, train_unlabelled, meta
