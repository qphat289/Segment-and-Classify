# import os
# import torch
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from metrics import create_metrics, DualTaskLoss
# from src.model import MobileNetUNet
# from dataset import get_data_loaders
# from trainer import ModelTrainer
# from pathlib import Path

# def main():
#     # Configuration
#     CONFIG = {
#         'data_dir': r'C:\Users\Xpeedent\Desktop\FPT\SP25\DBM\data_DBM\datatest',
#         'batch_size': 8,
#         'num_workers': 4,
#         'learning_rate': 0.0005,
#         'num_epochs': 150,
#         'patience': 30,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'seg_weight': 1.2,
#         'cls_weight': 0.4,                              
#         'save_dir': 'checkpoint'  
#     }

#     os.makedirs(CONFIG['save_dir'], exist_ok=True)
#     save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')

#     # 1. Initialize model
#     model = MobileNetUNet(                         
#         img_ch=1,
#         seg_ch=4,
#         num_classes=4
#     ).to(CONFIG['device'])

#     # 2. Create data loaders
#     train_loader, val_loader = get_data_loaders(
#         root_dir=CONFIG['data_dir'],
#         batch_size=CONFIG['batch_size'],
#         num_workers=CONFIG['num_workers']
#     )

#     # 3. Setup metrics
#     metrics = create_metrics()

#     # 4. Initialize optimizer and loss functions
#     optimizer = Adam(
#         model.parameters(), 
#         lr=CONFIG['learning_rate'],
#         weight_decay=5e-5,
#         betas=(0.9, 0.999)
#     )
    
#     criterion = DualTaskLoss(
#         seg_weight=CONFIG['seg_weight'],
#         cls_weight=CONFIG['cls_weight']
#     )
    
#     # 5. Initialize scheduler
#     scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode='max',
#         patience=10,
#         factor=0.5,
#         verbose=True,
#         min_lr=1e-6
#     )

#     # 6. Initialize trainer
#     trainer = ModelTrainer(
#         model=model,
#         dataloaders={
#             'train': train_loader,
#             'val': val_loader
#         },
#         criterion_seg=criterion.seg_criterion,
#         criterion_cls=criterion.cls_criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         metrics=metrics,
#         device=CONFIG['device'],
#         patience=CONFIG['patience'],
#         task_weights={
#             'seg': CONFIG['seg_weight'],
#             'cls': CONFIG['cls_weight']
#         }
#     )

#     # 7. Train the model
#     model = trainer.train(
#         num_epochs=CONFIG['num_epochs'],
#         save_path=save_path
#     )
    
#     # Save final model
#     final_save_path = os.path.join(CONFIG['save_dir'], 'final_model.pt')
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
#         'config': CONFIG
#     }, final_save_path)

#     print("Training completed successfully!")

# if __name__ == "__main__":
#     main()

import os
import torch
import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import create_metrics, DualTaskLoss
from src.model import MobileNetUNet
from dataset import get_data_loaders
from trainer import ModelTrainer
from pathlib import Path

def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Train a brain tumor segmentation and classification model.")
    
    # Configuration parameters
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of epochs to train.")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping.")
    parser.add_argument('--seg_weight', type=float, default=1.2, help="Weight for the segmentation loss.")
    parser.add_argument('--cls_weight', type=float, default=0.4, help="Weight for the classification loss.")
    parser.add_argument('--save_dir', type=str, default='checkpoint_new', help="Directory to save model checkpoints.")
    
    # Parse arguments
    args = parser.parse_args()

    # Configuration
    CONFIG = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seg_weight': args.seg_weight,
        'cls_weight': args.cls_weight,
        'save_dir': args.save_dir
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')

    # 1. Initialize model
    model = MobileNetUNet(                         
        img_ch=1,
        seg_ch=4,
        num_classes=4
    ).to(CONFIG['device'])

    # 2. Create data loaders
    train_loader, val_loader = get_data_loaders(
        root_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # 3. Setup metrics
    metrics = create_metrics()

    # 4. Initialize optimizer and loss functions
    optimizer = Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=5e-5,
        betas=(0.9, 0.999)
    )
    
    criterion = DualTaskLoss(
        seg_weight=CONFIG['seg_weight'],
        cls_weight=CONFIG['cls_weight']
    )
    
    # 5. Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10,
        factor=0.5,
        verbose=True,
        min_lr=1e-6
    )

    # 6. Initialize trainer
    trainer = ModelTrainer(
        model=model,
        dataloaders={
            'train': train_loader,
            'val': val_loader
        },
        criterion_seg=criterion.seg_criterion,
        criterion_cls=criterion.cls_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        task_weights={
            'seg': CONFIG['seg_weight'],
            'cls': CONFIG['cls_weight']
        }
    )

    # 7. Train the model
    model = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_path=save_path
    )
    
    # Save final model
    final_save_path = os.path.join(CONFIG['save_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': CONFIG
    }, final_save_path)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
