import config
from imports import *
from  dataset_classes import *


def create_transforms(image_size: int = 112, augment: bool = True) -> T.Compose:
    """Create transform pipeline for face recognition."""

    transform_list = [
        T.Resize((image_size, image_size)),

        T.ToTensor(),

        T.ToDtype(torch.float32, scale=True),
    ]

    if augment:  
        transform_list.extend([
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=10),        
        ])

    transform_list.extend([
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
    ])

    return T.Compose(transform_list)


train_transforms = create_transforms(augment=config['augument'])

val_transforms   = create_transforms(augment=False)



# Datasets
cls_train_dataset = ImageDataset(os.path.join(config['cls_data'], 'train'), train_transforms, config['num_classes'] )
cls_val_dataset   = ImageDataset(os.path.join(config['cls_data'], 'dev'), val_transforms, config['num_classes'])
cls_test_dataset  = ImageDataset(os.path.join(config['cls_data'], 'test'), val_transforms, config['num_classes'])

assert cls_train_dataset.classes == cls_val_dataset.classes == cls_test_dataset.classes, "Class mismatch!"


# Dataloaders
cls_train_loader = DataLoader(cls_train_dataset, batch_size=config['batch_size'], shuffle=True,  num_workers=4, pin_memory=True)
cls_val_loader   = DataLoader(cls_val_dataset,   batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
cls_test_loader  = DataLoader(cls_test_dataset,  batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)


# Verification Datasets
ver_val_dataset  = ImagePairDataset(config['ver_data_dir'], config['val_pairs_file'], val_transforms)
ver_test_dataset = TestImagePairDataset(config['ver_data_dir'], config["test_pairs_file"], val_transforms)

# verification Dataloader
ver_val_loader   = DataLoader(ver_val_dataset,  batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
ver_test_loader  = DataLoader(ver_test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
