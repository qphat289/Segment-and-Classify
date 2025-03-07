import torch
import torch.nn as nn

class SegmentationMetrics:
    def __init__(self, smooth=0.4):
        self.smooth = smooth

    # def dice_coef_metric(self, pred, label):
    #     # pred shape: [B, C, H, W], label shape: [B, H, W]
    #     pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    #     label_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
    #     label_one_hot.scatter_(1, label.unsqueeze(1), 1)  # Convert to one-hot encoding
        
    #     intersection = (pred * label_one_hot).sum(dim=(2, 3))  # Calculate intersection
    #     denominator = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))  # Calculate denominator
    #     dice = (2. * intersection + self.smooth) / (denominator + self.smooth)  # Calculate Dice coefficient
        
    #     return dice.mean()  # Average over batch and classes

    def dice_coef_metric(self, pred, label, ignore_background=True):
        # pred shape: [B, C, H, W], label shape: [B, H, W]
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
        label_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)  # Convert to one-hot encoding
        
        # Create class weights - by default all 1s
        num_classes = pred.size(1)
        class_weights = torch.ones(num_classes, device=pred.device)
        
        # Set background weight to 0 if ignoring background
        if ignore_background:
            class_weights[0] = 0
        
        # Calculate per-class dice scores
        intersection = (pred * label_one_hot).sum(dim=(2, 3))  # [B, C]
        denominator = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))  # [B, C]
        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)  # [B, C]
        
        # Apply class weights to each class score
        weighted_dice = dice_per_class * class_weights.view(1, -1)
        
        # Calculate mean over non-ignored classes
        num_valid_classes = (class_weights > 0).sum()
        if num_valid_classes > 0:
            return weighted_dice.sum() / (num_valid_classes * pred.size(0))
        else:
            return torch.tensor(1.0, device=pred.device)

    def iou(self, pred, label, ignore_classes=None):
        # pred shape: [B, C, H, W], label shape: [B, H, W]
        if ignore_classes is None:
            ignore_classes = [0]  # Default to ignoring background class
            
        pred = torch.softmax(pred, dim=1)
        label_one_hot = torch.zeros_like(pred)
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        
        # Calculate IoU for all classes
        intersection = (pred * label_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3)) - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)
        
        # Create class weights - ignore specified classes
        num_classes = pred.size(1)
        class_weights = torch.ones(num_classes, device=pred.device)
        for cls_idx in ignore_classes:
            if 0 <= cls_idx < num_classes:
                class_weights[cls_idx] = 0
        
        # Apply class weights
        weighted_iou = iou_per_class * class_weights.view(1, -1)
        
        # Calculate mean over non-ignored classes
        num_valid_classes = class_weights.sum()
        if num_valid_classes > 0:
            return weighted_iou.sum() / (num_valid_classes * pred.size(0))
        else:
            return torch.tensor(1.0, device=pred.device)



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
    def __init__(self, seg_weight=1.0, cls_weight=1.0, smooth=0.4):
        super(DualTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_criterion = self.dice_loss         # Use Dice Loss for segmentation
        self.cls_criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
        self.smooth = smooth

    def dice_loss(self, pred, target, ignore_background=True):
    # pred shape: [B, C, H, W], label shape: [B, H, W]
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
        label_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
        label_one_hot.scatter_(1, target.unsqueeze(1), 1)  # Convert to one-hot encoding
        
        # Create class weights - by default all 1s
        num_classes = pred.size(1)
        class_weights = torch.ones(num_classes, device=pred.device)
        
        # Set background weight to 0 if ignoring background
        if ignore_background:
            class_weights[0] = 0
        
        # Calculate per-class dice scores
        intersection = (pred * label_one_hot).sum(dim=(2, 3))  # [B, C]
        denominator = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))  # [B, C]
        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)  # [B, C]
        
        # Apply class weights to each class score
        weighted_dice = dice_per_class * class_weights.view(1, -1)
        
        # Calculate mean over non-ignored classes
        num_valid_classes = (class_weights > 0).sum()
        if num_valid_classes > 0:
            dice_coef =  weighted_dice.sum() / (num_valid_classes * pred.size(0))
        else:
            dice_coef = torch.tensor(1.0, device=pred.device)
        return 1.0 - dice_coef

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice_loss(seg_pred, seg_target)  # Calculate Dice Loss for segmentation
        cls_loss = self.cls_criterion(cls_pred, cls_target)  # Calculate CrossEntropyLoss for classification
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss  # Combine losses


def create_metrics():
    seg_metrics = SegmentationMetrics()
    cls_metrics = ClassificationMetrics()

    metrics = {
        'seg_dice': seg_metrics.dice_coef_metric,
        'seg_iou': seg_metrics.iou,
        'cls_accuracy': cls_metrics.accuracy,
    }
    
    return metrics



# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SegmentationMetrics:
#     def __init__(self, smooth=0.4):
#         self.smooth = smooth

#     def dice_coef_metric(self, pred, label, ignore_background=True):
#         """
#         Improved Dice coefficient calculation that properly handles background ignoring
        
#         Args:
#             pred: [B, C, H, W] - prediction logits
#             label: [B, H, W] - ground truth labels
#             ignore_background: if True, background class is excluded
            
#         Returns:
#             dice_mean: mean dice over included classes
#             class_dice: dictionary of per-class dice scores
#         """
#         # Apply softmax
#         pred_softmax = F.softmax(pred, dim=1)
        
#         # One-hot encode target
#         num_classes = pred.size(1)
#         target_one_hot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
#         # Determine which classes to include
#         class_range = range(1, num_classes) if ignore_background else range(num_classes)
        
#         # Calculate dice for each class
#         class_dice = {}
#         dice_sum = 0.0
#         valid_classes = 0
        
#         for cls in class_range:
#             # Extract predictions and targets for this class
#             pred_cls = pred_softmax[:, cls]
#             target_cls = target_one_hot[:, cls]
            
#             # Check if this class exists in the batch
#             has_class = target_cls.sum() > 0
            
#             if has_class:
#                 # Calculate intersection and union
#                 intersection = (pred_cls * target_cls).sum()
#                 pred_sum = pred_cls.sum()
#                 target_sum = target_cls.sum()
                
#                 # Calculate dice - add smooth factor to both numerator and denominator
#                 dice_cls = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
                
#                 # Store class dice and update sum
#                 class_dice[cls] = dice_cls.item()
#                 dice_sum += dice_cls
#                 valid_classes += 1
        
#         # Calculate mean dice over included classes
#         if valid_classes > 0:
#             dice_mean = dice_sum / valid_classes
#         else:
#             dice_mean = torch.tensor(0.0, device=pred.device)
            
#         return dice_mean, class_dice

#     def iou(self, pred, label, ignore_background=True):
#         """
#         Improved IoU calculation that properly handles background ignoring
        
#         Args:
#             pred: [B, C, H, W] - prediction logits
#             label: [B, H, W] - ground truth labels
#             ignore_background: if True, background class is excluded
            
#         Returns:
#             iou_mean: mean IoU over included classes
#             class_iou: dictionary of per-class IoU scores
#         """
#         # Apply softmax
#         pred_softmax = F.softmax(pred, dim=1)
        
#         # One-hot encode target
#         num_classes = pred.size(1)
#         target_one_hot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
#         # Determine which classes to include
#         class_range = range(1, num_classes) if ignore_background else range(num_classes)
        
#         # Calculate IoU for each class
#         class_iou = {}
#         iou_sum = 0.0
#         valid_classes = 0
        
#         for cls in class_range:
#             # Extract predictions and targets for this class
#             pred_cls = pred_softmax[:, cls]
#             target_cls = target_one_hot[:, cls]
            
#             # Check if this class exists in the batch
#             has_class = target_cls.sum() > 0
            
#             if has_class:
#                 # Calculate intersection and union
#                 intersection = (pred_cls * target_cls).sum()
#                 union = pred_cls.sum() + target_cls.sum() - intersection
                
#                 # Calculate IoU - add smooth factor to both numerator and denominator
#                 iou_cls = (intersection + self.smooth) / (union + self.smooth)
                
#                 # Store class IoU and update sum
#                 class_iou[cls] = iou_cls.item()
#                 iou_sum += iou_cls
#                 valid_classes += 1
        
#         # Calculate mean IoU over included classes
#         if valid_classes > 0:
#             iou_mean = iou_sum / valid_classes
#         else:
#             iou_mean = torch.tensor(0.0, device=pred.device)
            
#         return iou_mean, class_iou

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
    
# def create_metrics():
#     seg_metrics = SegmentationMetrics()
#     cls_metrics = ClassificationMetrics()
#     metrics = {
#         'seg_dice': lambda pred, target: seg_metrics.dice_coef_metric(pred, target, ignore_background=True)[0],
#         'seg_iou': lambda pred, target: seg_metrics.iou(pred, target, ignore_background=True)[0],
#         'cls_accuracy': cls_metrics.accuracy,
#     }
    
#     return metrics

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=0.4, ignore_background=True):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ignore_background = ignore_background
        
#     def forward(self, inputs, targets):
#         """
#         Improved Dice loss implementation that properly ignores background
        
#         Args:
#             inputs: [B, C, H, W] - prediction logits
#             targets: [B, H, W] - ground truth labels (class indices)
            
#         Returns:
#             dice_loss: scalar loss value
#         """
#         # Apply softmax to get class probabilities
#         inputs = F.softmax(inputs, dim=1)
        
#         # One-hot encode targets
#         targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
#         # Dimensions
#         batch_size = inputs.size(0)
#         num_classes = inputs.size(1)
        
#         # Choose which classes to include in loss calculation
#         class_indices = range(1, num_classes) if self.ignore_background else range(num_classes)
        
#         # Initialize dice scores
#         dice_scores = []
        
#         # Calculate dice for each class in the selected range
#         for cls in class_indices:
#             # Get predictions and targets for this class
#             pred_cls = inputs[:, cls]  # [B, H, W]
#             target_cls = targets_one_hot[:, cls]  # [B, H, W]
            
#             # Check if this class exists in the batch (for metrics reporting)
#             has_class = (target_cls.sum() > 0)
            
#             # Skip if class doesn't exist and we're calculating metrics
#             if not has_class and not self.training:
#                 continue
                
#             # Calculate intersection and union
#             intersection = (pred_cls * target_cls).sum()
#             pred_sum = pred_cls.sum()
#             target_sum = target_cls.sum()
            
#             # Calculate dice coefficient for this class - handle empty case safely
#             # Add small constant to avoid division by zero while maintaining gradients
#             denominator = pred_sum + target_sum
#             if denominator > 0 or self.training:
#                 dice_cls = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
#                 dice_scores.append(dice_cls)
        
#         # Average dice scores if we have any
#         if len(dice_scores) > 0:
#             dice_loss = 1.0 - torch.stack(dice_scores).mean()
#         else:
#             # Handle case where no foreground classes exist in batch
#             # Return zero loss with gradient connection to inputs
#             dice_loss = 0.0 * inputs.sum()
            
#         return dice_loss

# class CombinedLoss(nn.Module):
#     def __init__(self, seg_weight=1.0, cls_weight=0.5, ignore_background=True):
#         super(CombinedLoss, self).__init__()
#         self.seg_weight = seg_weight
#         self.cls_weight = cls_weight
#         self.seg_criterion = DiceLoss(ignore_background=ignore_background)
#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100 if ignore_background else -1)
#         self.cls_criterion = nn.CrossEntropyLoss()
        
#     def forward(self, seg_pred, seg_target, cls_pred=None, cls_target=None):
#         """
#         Calculate combined loss for segmentation and classification
        
#         Args:
#             seg_pred: [B, C, H, W] - segmentation prediction logits
#             seg_target: [B, H, W] - segmentation ground truth labels
#             cls_pred: [B, C] - classification prediction logits (optional)
#             cls_target: [B] - classification ground truth labels (optional)
            
#         Returns:
#             loss: combined weighted loss
#         """
#         # Calculate segmentation losses - combine Dice and CE
#         dice_loss = self.dice_loss(seg_pred, seg_target)
        
#         # For CE loss, we might want to mask out background pixels
#         ce_loss = self.ce_loss(seg_pred, seg_target)
        
#         # Combine segmentation losses
#         seg_loss = dice_loss + ce_loss
        
#         # Add classification loss if provided
#         if cls_pred is not None and cls_target is not None:
#             cls_loss = self.cls_criterion(cls_pred, cls_target)
#             total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
#         else:
#             total_loss = seg_loss
            
#         return total_loss

