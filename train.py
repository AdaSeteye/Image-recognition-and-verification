from architecture import Architecture
import config
from imports import torch
from utils import *
from loading_data import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Architecture(num_classes=config['num_classes']).to(DEVICE)



criterion = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55)

scaler = torch.cuda.amp.GradScaler()



e = 0
best_valid_cls_acc = 0.0
eval_cls = True
best_valid_ret_acc = 0.0

for epoch in range(e, config['epochs']):
    # epoch
    print("\nEpoch {}/{}".format(epoch + 1, config['epochs']))

    # train
    train_cls_acc, train_loss = train_epoch(model, cls_train_loader, optimizer, scheduler, scaler, DEVICE, config)
    curr_lr = float(optimizer.param_groups[0]['lr'])
    print("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr
    ))
    
    metrics = {
        'train_cls_acc': train_cls_acc,
        'train_loss': train_loss,
    }

    if eval_cls:
        valid_cls_acc, valid_loss = valid_epoch_cls(model, cls_val_loader, DEVICE, config)
        print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
        metrics.update({
            'valid_cls_acc': valid_cls_acc,
            'valid_loss': valid_loss,
        })

    valid_ret_acc = valid_epoch_ver(model, ver_val_loader, DEVICE, config)
    print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
    metrics.update({'valid_ret_acc': valid_ret_acc})

    save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
    print("Saved epoch model")

    if eval_cls and valid_cls_acc >= best_valid_cls_acc:
        best_valid_cls_acc = valid_cls_acc
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
        wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
        print("Saved best classification model")

    if valid_ret_acc >= best_valid_ret_acc:
        best_valid_ret_acc = valid_ret_acc
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
        wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
        print("Saved best retrieval model")


    

