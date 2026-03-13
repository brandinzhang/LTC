import pickle
import torch

from torchvision import transforms
from data.cub import get_cub_datasets
from data.scars import get_scars_datasets
from data.food101 import get_food_101_datasets
from data.pets import get_oxford_pets_datasets
from data.inaturalist import get_inaturalist_datasets
from data.cifar10 import get_cifar10_datasets
from data.cifar100 import get_cifar100_datasets
from data.imagenet100_merged import get_imagenet100_merged_datasets



from copy import deepcopy


def build_dataset(args, train_transform=None, test_transform=None): 

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])      

    if args.data_set == 'cub':

        split_path = 'data/ssb_splits/cub_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled = get_cub_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=args.prop_train_labels, data_root=args.data_root)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=100
        args.unlabeled_nums=200
        args.known_classes = train_classes

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'food':

        train_dataset, test_dataset, train_dataset_unlabelled = get_food_101_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(51), prop_train_labels=args.prop_train_labels, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=51
        args.unlabeled_nums=101
        args.known_classes = range(51)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'pets':

        train_dataset, test_dataset, train_dataset_unlabelled = get_oxford_pets_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(19), prop_train_labels=args.prop_train_labels, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=19
        args.unlabeled_nums=38
        args.known_classes = range(19)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'Animalia':

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Animalia',
                                   train_classes=range(39), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=39
        args.unlabeled_nums=77
        args.known_classes = range(39)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'Arachnida':

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Arachnida',
                                   train_classes=range(28), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=28
        args.unlabeled_nums=56
        args.known_classes = range(28)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'Fungi':

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Fungi',
                                   train_classes=range(61), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=61
        args.unlabeled_nums=121
        args.known_classes = range(61)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'Mollusca':

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Mollusca',
                                   train_classes=range(47), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=47
        args.unlabeled_nums=93
        args.known_classes = range(47)

        return train_dataset, test_dataset, unlabelled_train_examples_test


    elif args.data_set == 'scars':

        split_path = 'data/ssb_splits/scars_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled = get_scars_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=args.prop_train_labels, data_root=args.data_root)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums=98
        args.unlabeled_nums=196
        args.known_classes = train_classes

        return train_dataset, test_dataset, unlabelled_train_examples_test
    elif args.data_set == 'cifar10':

        # By default, the first 5 classes are treated as known (consistent with the dataset style used here)
        train_dataset, test_dataset, train_dataset_unlabelled = get_cifar10_datasets(
            train_transform=transform,
            test_transform=test_transform,
            train_classes=range(5),                 # 5 known
            prop_train_labels=args.prop_train_labels,
            data_root=args.data_root
        )

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums = 5
        args.unlabeled_nums = 10
        args.known_classes = range(5)

        return train_dataset, test_dataset, unlabelled_train_examples_test

    elif args.data_set == 'cifar100':

        
        train_dataset, test_dataset, train_dataset_unlabelled = get_cifar100_datasets(
            train_transform=transform,
            test_transform=test_transform,
            train_classes=range(80),                # 80 known
            prop_train_labels=args.prop_train_labels,
            data_root=args.data_root
        )

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        args.labeled_nums = 80
        args.unlabeled_nums = 100
        args.known_classes = range(80)

        return train_dataset, test_dataset, unlabelled_train_examples_test
    elif args.data_set == 'imagenet100m':
        split_path = 'data/ssb_splits/imagenet100m_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        known_classes = class_info['known_classes']                         # e.g., 50 integers in [0..99]
        open_set_classes = class_info['unknown_classes']                    # dict: Easy/Medium/Hard -> list[int]
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled, meta = \
            get_imagenet100_merged_datasets(
                train_transform=transform,
                test_transform=test_transform,
                data_root=args.data_root,
                known_count=len(known_classes),
                known_class_list=known_classes,             # known set specified by the pkl split
                prop_train_labels=args.prop_train_labels,
                seed=args.seed
            )

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform

        args.labeled_nums = len(known_classes)   # 50
        args.unlabeled_nums = 100                # total number of classes (ImageNet-100)
        args.known_classes = known_classes

        print("train_classes:", known_classes)
        print("len(train_classes):", len(known_classes))
        print("unlabeled_classes (concat E/M/H):", len(unlabeled_classes))
        print("len(unlabeled_classes):", len(unlabeled_classes))

        return train_dataset, test_dataset, unlabelled_train_examples_test
