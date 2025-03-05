# import torch
# import torch.nn as nn

# class SegmentationMetrics:
#     def __init__(self, smooth=1.0, threshold=0.5):
#         self.smooth = smooth
#         self.threshold = threshold

#     def dice_coef_metric(self, pred, label):
#         # pred shape: [B, C, H, W], label shape: [B, H, W]
#         pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
#         label_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
#         label_one_hot.scatter_(1, label.unsqueeze(1), 1)  # Convert to one-hot encoding
        
#         intersection = (pred * label_one_hot).sum(dim=(2, 3))  # Calculate intersection
#         denominator  = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))  # Calculate denominator
#         dice = (2. * intersection + self.smooth) / (denominator + self.smooth)  # Calculate Dice coefficient
        
#         return dice.mean()  # Average over batch and classes

#     def iou(self, pred, label):
#         pred = torch.softmax(pred, dim=1)
#         label_one_hot = torch.zeros_like(pred)
#         label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        
#         intersection = (pred * label_one_hot).sum(dim=(2, 3))
#         union = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3)) - intersection
#         iou = (intersection + self.smooth) / (union + self.smooth)
        
#         return iou.mean()


# class ClassificationMetrics:
#     def accuracy(self, pred, label):
#         pred_cls = torch.argmax(pred, dim=1)
#         accuracy = (pred_cls == label).float().mean()
#         return accuracy * 100

#     def f1_score_cls(self, pred, label):
#         pred_cls = torch.argmax(pred, dim=1)
#         f1_scores = []
        
#         for cls_idx in range(pred.size(1)):
#             true_pos = ((pred_cls == cls_idx) & (label == cls_idx)).sum().float()
#             false_pos = ((pred_cls == cls_idx) & (label != cls_idx)).sum().float()
#             false_neg = ((pred_cls != cls_idx) & (label == cls_idx)).sum().float()
            
#             precision = true_pos / (true_pos + false_pos + 1e-6)
#             recall = true_pos / (true_pos + false_neg + 1e-6)
            
#             f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
#             f1_scores.append(f1)
        
#         return torch.stack(f1_scores).mean()


# class DualTaskLoss(nn.Module):
#     def __init__(self, seg_weight=1.0, cls_weight=1.0):
#         super(DualTaskLoss, self).__init__()
#         self.seg_weight = seg_weight
#         self.cls_weight = cls_weight
#         self.seg_criterion = self.dice_loss         # Use Dice Loss for segmentation
#         self.cls_criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

#     def dice_loss(self, pred, target):
#         pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
#         target_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
#         target_one_hot.scatter_(1, target.unsqueeze(1), 1)  # Convert to one-hot encoding
        
#         intersection = (pred * target_one_hot).sum(dim=(2, 3))  # Calculate intersection
#         denominator = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # Calculate denominator
#         dice = (2. * intersection + 1e-6) / (denominator + 1e-6)  # Calculate Dice coefficient
        
#         return 1 - dice.mean()  # Return Dice Loss (1 - Dice Coefficient)

#     def forward(self, seg_pred, seg_target, cls_pred, cls_target):
#         seg_loss = self.dice_loss(seg_pred, seg_target)  # Calculate Dice Loss for segmentation
#         cls_loss = self.cls_criterion(cls_pred, cls_target)  # Calculate CrossEntropyLoss for classification
#         return self.seg_weight * seg_loss + self.cls_weight * cls_loss  # Combine losses


# def create_metrics():
#     seg_metrics = SegmentationMetrics()
#     cls_metrics = ClassificationMetrics()

#     metrics = {
#         'seg_dice': seg_metrics.dice_coef_metric,
#         'seg_iou': seg_metrics.iou,
#         'cls_accuracy': cls_metrics.accuracy,
#     }
    
#     return metrics



import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, smooth=1.0, threshold=0.5, background_idx=0):
        self.smooth = smooth
        self.threshold = threshold
        self.background_idx = background_idx

    def dice_coef_metric(self, pred, label, ignore_background=True):
        """
        Calculate Dice coefficient with option to ignore background
        
        Args:
            pred: [B, C, H, W] - prediction logits
            label: [B, H, W] - ground truth labels
            ignore_background: if True, background class is excluded from average
        """
        pred = torch.softmax(pred, dim=1)
        label_one_hot = torch.zeros_like(pred)
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        
        batch_size, num_classes = pred.shape[0], pred.shape[1]
        
        # Calculate Dice for each class and each sample in batch
        intersection = (pred * label_one_hot).sum(dim=(2, 3))
        denominator = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))
        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        if ignore_background:
            # Exclude background class (assumed to be index 0)
            # Only average over tumor classes (indices 1, 2, 3)
            dice_per_class = dice_per_class[:, 1:]
            
            # Only include classes that actually appear in the ground truth
            # to avoid division by zero or meaningless metrics
            valid_mask = label_one_hot[:, 1:].sum(dim=(2, 3)) > 0
            
            # Calculate mean only over valid classes
            dice_valid = dice_per_class * valid_mask.float()
            dice_mean = dice_valid.sum() / (valid_mask.sum() + 1e-6)
            
            # Also calculate per-class average for reporting
            class_dice = {}
            for c in range(1, num_classes):  # Skip background
                class_valid = valid_mask[:, c-1]  # Adjust index since we removed background
                if class_valid.sum() > 0:
                    class_dice[c] = (dice_per_class[:, c-1] * class_valid.float()).sum() / class_valid.sum()
                else:
                    class_dice[c] = torch.tensor(0.0, device=pred.device)
                    
            return dice_mean, class_dice
        else:
            # Return mean over all classes including background
            return dice_per_class.mean(), {c: dice_per_class[:, c].mean() for c in range(num_classes)}

    def iou(self, pred, label, ignore_background=True):
        """
        Calculate IoU with option to ignore background
        """
        pred = torch.softmax(pred, dim=1)
        label_one_hot = torch.zeros_like(pred)
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        
        batch_size, num_classes = pred.shape[0], pred.shape[1]
        
        # Calculate IoU for each class and each sample in batch
        intersection = (pred * label_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3)) - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)
        
        if ignore_background:
            # Exclude background class (assumed to be index 0)
            iou_per_class = iou_per_class[:, 1:]
            
            # Only include classes that actually appear in the ground truth
            valid_mask = label_one_hot[:, 1:].sum(dim=(2, 3)) > 0
            
            # Calculate mean only over valid classes
            iou_valid = iou_per_class * valid_mask.float()
            iou_mean = iou_valid.sum() / (valid_mask.sum() + 1e-6)
            
            # Also calculate per-class average for reporting
            class_iou = {}
            for c in range(1, num_classes):  # Skip background
                class_valid = valid_mask[:, c-1]  # Adjust index since we removed background
                if class_valid.sum() > 0:
                    class_iou[c] = (iou_per_class[:, c-1] * class_valid.float()).sum() / class_valid.sum()
                else:
                    class_iou[c] = torch.tensor(0.0, device=pred.device)
                    
            return iou_mean, class_iou
        else:
            # Return mean over all classes including background
            return iou_per_class.mean(), {c: iou_per_class[:, c].mean() for c in range(num_classes)}

class ClassificationMetrics:
    def accuracy(self, pred, label):
        pred_cls = torch.argmax(pred, dim=1)
        accuracy = (pred_cls == label).float().mean()
        return accuracy * 100

    def f1_score_cls(self, pred, label):
        pred_cls = torch.argmax(pred, dim=1)
        f1_scores = []
        
        for cls_idx in range(pred.size(1)):
            true_pos = ((pred_cls == cls_idx) & (label == cls_idx)).sum().float()
            false_pos = ((pred_cls == cls_idx) & (label != cls_idx)).sum().float()
            false_neg = ((pred_cls != cls_idx) & (label == cls_idx)).sum().float()
            
            precision = true_pos / (true_pos + false_pos + 1e-6)
            recall = true_pos / (true_pos + false_neg + 1e-6)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_scores.append(f1)
        
        return torch.stack(f1_scores).mean()

class DualTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=1.0, background_idx=0):
        super(DualTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.background_idx = background_idx
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = self.dice_loss_ignore_background
        
    def dice_loss_ignore_background(self, pred, target):
        """
        Dice loss that ignores background class
        
        Args:
            pred: [B, C, H, W] - prediction logits
            target: [B, H, W] - ground truth labels
        """
        # Convert to probabilities
        pred_softmax = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = torch.zeros_like(pred_softmax)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate Dice Loss for each class separately
        batch_size, num_classes = pred.shape[0], pred.shape[1]
        dice_loss = 0
        num_fg_classes = 0  # Counter for foreground classes
        
        # Loop through all classes except background
        for cls in range(1, num_classes):  # Skip background (index 0)
            # Check if this class exists in the batch
            has_class = (target == cls).sum() > 0
            
            if has_class:
                pred_cls = pred_softmax[:, cls]
                target_cls = target_one_hot[:, cls]
                
                intersection = (pred_cls * target_cls).sum()
                denominator = pred_cls.sum() + target_cls.sum()
                
                # Calculate Dice for this class
                dice_cls = (2. * intersection + 1e-6) / (denominator + 1e-6)
                dice_loss += (1 - dice_cls)
                num_fg_classes += 1
        
        # Average Dice loss over foreground classes
        if num_fg_classes > 0:
            dice_loss = dice_loss / num_fg_classes
        else:
            # If no foreground classes in batch, set loss to 0
            dice_loss = torch.tensor(0.0, device=pred.device)
            
        return dice_loss

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice_loss_ignore_background(seg_pred, seg_target)
        cls_loss = self.cls_criterion(cls_pred, cls_target)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss

def create_metrics(background_idx=0):
    seg_metrics = SegmentationMetrics(background_idx=background_idx)
    cls_metrics = ClassificationMetrics()

    metrics = {
        'seg_dice': lambda pred, target: seg_metrics.dice_coef_metric(pred, target, ignore_background=True)[0],
        'seg_iou': lambda pred, target: seg_metrics.iou(pred, target, ignore_background=True)[0],
        'cls_accuracy': cls_metrics.accuracy,
    }
    
    return metrics

