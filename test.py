import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model import MobileNetUNet
import torchvision.transforms as transforms
from skimage import measure
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
from skimage.measure import approximate_polygon
from scipy.interpolate import splprep, splev
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def load_model(model_path, device):
    """Load model from checkpoint"""
    print(f"Using device: {device}")
    
    # Load model
    model = MobileNetUNet(img_ch=1, seg_ch=4, num_classes=4).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model

def load_and_preprocess_images(image_path, mask_path):
    """Load and preprocess image and mask"""
    # Load image
    img = Image.open(image_path)
    if img.mode != 'L':  # If not grayscale
        img = img.convert('L')  # Convert to grayscale
    
    # Load ground truth mask
    gt_mask = Image.open(mask_path)
    if gt_mask.mode != 'L':
        gt_mask = gt_mask.convert('L')

    # Resize to match training data size
    img = img.resize((256, 256))
    gt_mask = gt_mask.resize((256, 256))
    
    # Convert to numpy for visualization
    img_np = np.array(img)
    gt_mask_np = np.array(gt_mask)
    
    # Normalize ground truth mask if needed
    if gt_mask_np.max() > 3:  # If values are not 0-3
        gt_mask_np = (gt_mask_np / 255 * 3).astype(np.uint8)
    
    return img, gt_mask, img_np, gt_mask_np

def get_model_prediction(model, img, device):
    """Get model prediction for an image"""
    # Convert to tensor for model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        seg_output, cls_output = model(img_tensor)
        
        # Get raw segmentation output for dice calculation
        seg_output_np = seg_output.squeeze().cpu().numpy()  # [C, H, W]
        
        # Process segmentation output for visualization
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
        
        # Process classification output
        cls_pred = torch.argmax(cls_output, dim=1).item()
        cls_probs = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()
    
    return seg_output_np, seg_pred, cls_pred, cls_probs

def softmax(x, axis=0):
    """
    Compute softmax values for the given array along specified axis
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def dice_coef_metric(pred_mask, gt_mask, cls_pred, smooth=1.0, ignore_background=True):
    """
    Calculate Dice coefficient between predicted and ground truth masks
    for the predicted tumor class only, with option to ignore background
    
    Args:
        pred_mask: Predicted segmentation mask with shape [H, W] containing class indices
        gt_mask: Ground truth mask with shape [H, W] containing class indices
        cls_pred: Predicted class (0: No Tumor, 1: Glioma, 2: Meningioma, 3: Pituitary)
        smooth: Smoothing factor to prevent division by zero
        ignore_background: If True, background (class 0) will be ignored in calculations
    
    Returns:
        Dice coefficient for the predicted class
    """
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # For "No Tumor" prediction (cls_pred = 0)
    if cls_pred == 0:
        if ignore_background:
            # When ignoring background and prediction is "No Tumor",
            # we check if ground truth has any foreground classes
            has_foreground_in_gt = np.any(gt_mask > 0)
            if not has_foreground_in_gt:
                return 1.0  # Perfect match - neither has tumor
            else:
                return 0.0  # Ground truth has tumor but prediction doesn't
        else:
            # Original behavior when not ignoring background
            if np.all(gt_mask == 0):
                return 1.0  # Perfect match
            else:
                # Compare background regions
                pred_bg = (pred_mask == 0)
                gt_bg = (gt_mask == 0)
                intersection = np.sum(pred_bg & gt_bg)
                denominator = np.sum(pred_bg) + np.sum(gt_bg)
                return (2. * intersection + smooth) / (denominator + smooth)
    
    # For tumor classes (cls_pred = 1, 2, or 3)
    else:
        # Create binary masks for predicted class
        pred_c = (pred_mask == cls_pred)
        gt_c = (gt_mask == cls_pred)
        
        # Calculate intersection and union
        intersection = np.sum(pred_c & gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        
        # Calculate Dice
        if denominator > 0:
            return (2. * intersection + smooth) / (denominator + smooth)
        else:
            # If denominator is 0, it means either prediction or ground truth has no pixels
            # for this class
            if np.sum(gt_c) == 0:
                # If ground truth doesn't have this class but prediction does
                return 0.0
            else:
                # This case shouldn't happen (pred_c is empty but denominator is 0)
                return 0.0

def dice_coef_metric_agnostic(pred_mask, gt_mask, smooth=1.0):
    """Calculate Dice coefficient between any tumor regions, regardless of class"""
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Create binary masks for any tumor (classes 1, 2, or 3)
    pred_tumor = (pred_mask > 0)
    gt_tumor = (gt_mask > 0)
    
    # Calculate intersection and union
    intersection = np.sum(pred_tumor & gt_tumor)
    denominator = np.sum(pred_tumor) + np.sum(gt_tumor)
    
    # Calculate Dice
    if denominator > 0:
        return (2. * intersection + smooth) / (denominator + smooth)
    else:
        return 1.0 if np.sum(gt_tumor) == 0 else 0.0

def print_class_statistics(pred_mask, gt_mask):
    """Print statistics about class distribution in masks"""
    print("\nClass statistics:")
    print("Class | GT pixels | Pred pixels | Overlap")
    print("-" * 45)
    
    for c in range(4):  # Classes 0-3
        gt_c = (gt_mask == c)
        pred_c = (pred_mask == c)
        overlap = np.sum(gt_c & pred_c)
        print(f"{c}     | {np.sum(gt_c):9d} | {np.sum(pred_c):10d} | {overlap:7d}")
    
    # Calculate total overlap
    total_overlap = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size
    print(f"\nTotal pixel accuracy: {total_overlap / total_pixels:.4f}")

def smooth_contour(contour, tolerance=1.0, smoothing=True):
    """Simplify and smooth contours using spline interpolation"""
    # First simplify to reduce points
    simplified = approximate_polygon(contour, tolerance=tolerance)
    
    if smoothing and len(simplified) > 3:
        try:
            # Use spline interpolation for smoothing
            x, y = simplified[:, 1], simplified[:, 0]
            
            # Close the curve if it's not already closed
            if not np.array_equal(simplified[0], simplified[-1]):
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            
            # Fit spline
            tck, u = splprep([x, y], s=0, per=True)
            
            # Evaluate spline at more points for smoother curve
            u_new = np.linspace(0, 1, len(simplified) * 5)
            x_new, y_new = splev(u_new, tck)
            
            # Return as array
            return np.column_stack([y_new, x_new])
        except:
            # If spline fitting fails, return simplified contour
            return simplified
    else:
        return simplified

def plot_mask_and_contours(img_np, mask, color, sigma=0.2, threshold=0.7, 
                          tolerance=0.1, opacity=0.2, line_alpha=0.4):
    """Plot mask overlay and contour for a single mask"""
    # Clean up mask before finding contours
    cleaned_mask = binary_closing(binary_opening(mask, iterations=1), iterations=1)
    
    # Add filled overlay with low opacity
    overlay = np.zeros((*img_np.shape, 4))
    overlay[cleaned_mask] = [*color, opacity]  # Color with specified opacity
    plt.imshow(overlay)
    
    # Smooth mask for contours
    smoothed_mask = gaussian_filter(cleaned_mask.astype(float), sigma=sigma)
    
    # Add smoothed contour lines
    contours = measure.find_contours(smoothed_mask, threshold)
    for contour in contours:
        if len(contour) > 5:  # Only process contours with enough points
            smoothed = smooth_contour(contour, tolerance=tolerance, smoothing=True)
            plt.plot(smoothed[:, 1], smoothed[:, 0], color=color, linestyle='--', 
                     dashes=(5, 2), linewidth=1, alpha=line_alpha)

def create_visualization(img_np, gt_mask_np, seg_pred, cls_pred, cls_probs, avg_dice, class_agnostic_dice, output_dir):
    """Create and save visualization of results"""
    # Define class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Display original image
    plt.imshow(img_np, cmap='gray')
    
    # Plot ground truth mask and contours
    for i in range(1, 4):  # For each class (skipping background)
        gt_mask_class = (gt_mask_np == i)
        if np.any(gt_mask_class):
            plot_mask_and_contours(
                img_np, gt_mask_class, 
                color=(0, 1, 1),  # Cyan
                sigma=0.2, 
                threshold=0.7,
                tolerance=0.1, 
                opacity=0.2, 
                line_alpha=0.4
            )
    
    # Plot prediction mask and contours
    for i in range(1, 4):  # For each class (skipping background)
        pred_mask_class = (seg_pred == i)
        if np.any(pred_mask_class):
            plot_mask_and_contours(
                img_np, pred_mask_class, 
                color=(1, 0.5, 0),  # Orange
                sigma=0.2, 
                threshold=0.7,
                tolerance=0.1, 
                opacity=0.2, 
                line_alpha=0.4
            )
    
    # Add legend
    legend_elements = [
        Patch(facecolor='cyan', alpha=0.5, label='Ground Truth (Fill)'),
        Line2D([0], [0], color='cyan', linestyle='--', lw=1, alpha=0.9, label='Ground Truth (Contour)'),
        Patch(facecolor='orange', alpha=0.5, label='Prediction (Fill)'),
        Line2D([0], [0], color='orange', linestyle='--', lw=1, alpha=0.9, label='Prediction (Contour)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title with dice and classification results
    plt.title(f'Dice: {avg_dice:.4f} (Agnostic: {class_agnostic_dice:.4f}) | Class: {class_names[cls_pred]} ({cls_probs[cls_pred]*100:.1f}%)', 
              fontsize=14)
    
    plt.axis('off')
    
    # Save and show result
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_result.png'), dpi=300)
    plt.show()
    
    # Print results
    print(f"Predicted class: {class_names[cls_pred]} ({cls_probs[cls_pred]*100:.1f}%)")
    print(f"Class-specific Dice Score: {avg_dice:.4f}")
    print(f"Class-agnostic Dice Score: {class_agnostic_dice:.4f}")
    print(f"Result saved to: {os.path.join(output_dir, 'comparison_result.png')}")

def compare_prediction_with_gt(model_path, image_path, mask_path, device='cuda'):
    """
    Compare model prediction with ground truth mask using different colors with reduced opacity
    and smooth dashed contour lines
    
    Args:
        model_path: Path to the saved model (.pt file)
        image_path: Path to the image to test
        mask_path: Path to the ground truth mask
        device: Device to run inference on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, device)
    
    # Load and preprocess images
    img, gt_mask, img_np, gt_mask_np = load_and_preprocess_images(image_path, mask_path)
    
    # Get model prediction
    seg_output_np, seg_pred, cls_pred, cls_probs = get_model_prediction(model, img, device)
    
    # Print unique values in masks to debug
    print("Ground truth classes present:", np.unique(gt_mask_np))
    print("Predicted classes present:", np.unique(seg_pred))
    
    # Print detailed class statistics
    print_class_statistics(seg_pred, gt_mask_np)
    
    # Calculate Dice scores
    class_specific_dice = dice_coef_metric(seg_pred, gt_mask_np, cls_pred)
    class_agnostic_dice = dice_coef_metric_agnostic(seg_pred, gt_mask_np)
    
    print(f"Class-specific Dice: {class_specific_dice:.4f}")
    print(f"Class-agnostic Dice: {class_agnostic_dice:.4f}")
    
    # Create output directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    create_visualization(img_np, gt_mask_np, seg_pred, cls_pred, cls_probs, class_specific_dice, class_agnostic_dice, output_dir)
    
    # Print additional information for No Tumor cases
    if np.all(gt_mask_np == 0):
        print("This is a 'No Tumor' image.")
        non_bg_pixels = np.sum(seg_pred > 0)
        total_pixels = seg_pred.size
        print(f"Percentage of pixels predicted as tumor: {non_bg_pixels/total_pixels*100:.2f}%")
        if class_specific_dice == 1.0:
            print("Perfect score! Model correctly predicted 'No Tumor' with minimal false positives.")


if __name__ == "__main__":
    # Path to your trained model
    model_path = "checkpoint_ori/best_model.pt"
    
    # Path to test image and ground truth mask
    image_path = "test/2.jpg"
    mask_path = "test/2.png"
    
    # Run comparison
    compare_prediction_with_gt(model_path, image_path, mask_path)
