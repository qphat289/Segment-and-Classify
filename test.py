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
from pathlib import Path
import time

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

def dice_coef_metric(pred_mask, gt_mask, smooth=1.0, ignore_background=True):
    """
    Calculate Dice coefficient between any tumor regions, regardless of class.
    Ignores background (class 0) and treats all other classes as tumor tissue.
    
    Args:
        pred_mask: Predicted segmentation mask (class indices)
        gt_mask: Ground truth segmentation mask (class indices)
        smooth: Smoothing factor to avoid division by zero
        ignore_background: Whether to ignore background class (default: True)
        
    Returns:
        Dice coefficient score (float)
    """
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Create binary masks for any tumor (any non-zero class)
    if ignore_background:
        pred_tumor = (pred_mask > 0)  # Any non-background class
        gt_tumor = (gt_mask > 0)      # Any non-background class
    else:
        # If not ignoring background, consider all pixels
        pred_tumor = np.ones_like(pred_mask, dtype=bool)
        gt_tumor = np.ones_like(gt_mask, dtype=bool)
    
    # Calculate intersection and union
    intersection = np.sum(pred_tumor & gt_tumor)
    pred_area = np.sum(pred_tumor)
    gt_area = np.sum(gt_tumor)
    
    # Calculate Dice coefficient
    denominator = pred_area + gt_area
    
    # Handle edge cases
    if denominator == 0:
        # Both prediction and ground truth have no foreground pixels
        return 1.0
    elif gt_area == 0:
        # Ground truth has no foreground but prediction does
        return 0.0
    elif pred_area == 0:
        # Prediction has no foreground but ground truth does
        return 0.0
    else:
        # Normal case
        return (2. * intersection + smooth) / (denominator + smooth)

def iou_metric(pred_mask, gt_mask, smooth=1.0, ignore_background=True):
    """
    Calculate IoU (Intersection over Union) for tumor regions
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth mask
        smooth: Smoothing factor to avoid division by zero
        ignore_background: Whether to ignore background
        
    Returns:
        IoU score
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Create binary masks for any tumor
    if ignore_background:
        pred_tumor = (pred_mask > 0)
        gt_tumor = (gt_mask > 0)
    else:
        pred_tumor = np.ones_like(pred_mask, dtype=bool)
        gt_tumor = np.ones_like(gt_mask, dtype=bool)
    
    # Calculate intersection and union
    intersection = np.sum(pred_tumor & gt_tumor)
    union = np.sum(pred_tumor | gt_tumor)
    
    # Calculate IoU
    if union == 0:
        return 1.0 if np.sum(gt_tumor) == 0 else 0.0
    else:
        return (intersection + smooth) / (union + smooth)

def print_class_statistics(pred_mask, gt_mask):
    """Print statistics about class distribution in masks"""
    class_names = ['No Tumor/Background', 'Glioma', 'Meningioma', 'Pituitary']
    
    print("\nClass statistics:")
    print("| Class | Name | GT pixels | Pred pixels | Overlap | Dice |")
    print("|-------|------|-----------|-------------|---------|------|")
    
    for c in range(4):  # Classes 0-3
        gt_c = (gt_mask == c)
        pred_c = (pred_mask == c)
        overlap = np.sum(gt_c & pred_c)
        
        # Calculate per-class dice
        denominator = np.sum(gt_c) + np.sum(pred_c)
        if denominator > 0:
            dice = (2.0 * overlap) / denominator
        else:
            dice = 1.0 if np.sum(gt_c) == 0 else 0.0
        
        print(f"| {c} | {class_names[c]} | {np.sum(gt_c):9d} | {np.sum(pred_c):11d} | {overlap:7d} | {dice:.4f} |")
    
    # Calculate total overlap
    total_overlap = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size
    print(f"\nPixel Accuracy: {total_overlap / total_pixels:.4f}")

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

def create_combined_visualization(img_np, gt_mask_np, seg_pred, cls_pred, cls_probs, 
                                 dice_score, iou_score, output_dir, filename_base):
    """
    Create visualization showing both ground truth and prediction on the same image
    with both Dice and IoU metrics, and colored insight zones
    """
    # Define class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Display original image
    plt.imshow(img_np, cmap='gray')
    
    # Create binary masks
    gt_tumor = gt_mask_np > 0
    pred_tumor = seg_pred > 0
    
    # Add colored insight zones with very low opacity
    # Ground truth - cyan fill with very low opacity
    if np.any(gt_tumor):
        gt_overlay = np.zeros((*img_np.shape, 4))
        gt_overlay[gt_tumor] = [0, 1, 1, 0.15]  # Cyan with very low opacity
        plt.imshow(gt_overlay)
    
    # Prediction - orange fill with very low opacity
    if np.any(pred_tumor):
        pred_overlay = np.zeros((*img_np.shape, 4))
        pred_overlay[pred_tumor] = [1, 0.5, 0, 0.15]  # Orange with very low opacity
        plt.imshow(pred_overlay)
    
    # Plot ground truth contour - cyan dashed line
    if np.any(gt_tumor):
        contours = measure.find_contours(gt_tumor.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 5:
                smoothed = smooth_contour(contour, tolerance=0.1, smoothing=True)
                plt.plot(smoothed[:, 1], smoothed[:, 0], color=(0, 1, 1), 
                        linestyle='--', linewidth=2, alpha=0.8)
    
    # Plot prediction contour - orange solid line
    if np.any(pred_tumor):
        contours = measure.find_contours(pred_tumor.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 5:
                smoothed = smooth_contour(contour, tolerance=0.1, smoothing=True)
                plt.plot(smoothed[:, 1], smoothed[:, 0], color=(1, 0.5, 0), 
                        linewidth=2, alpha=0.8)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=(0, 1, 1), alpha=0.3, label='Ground Truth'),
        Patch(facecolor=(1, 0.5, 0), alpha=0.3, label='Prediction'),
        Line2D([0], [0], color=(0, 1, 1), linestyle='--', lw=2, alpha=0.8, label='Ground Truth Contour'),
        Line2D([0], [0], color=(1, 0.5, 0), lw=2, alpha=0.8, label='Prediction Contour')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add metrics and prediction class
    title = f"Dice: {dice_score:.4f} | IoU: {iou_score:.4f}\nPredicted: {class_names[cls_pred]} ({cls_probs[cls_pred]*100:.1f}%)"
    plt.title(title, fontsize=14)
    
    plt.axis('off')
    
    # Save and show result
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{filename_base}_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return path for reference
    return output_path


def compare_prediction_with_gt(model_path, image_path, mask_path, output_dir='results', device='cuda'):
    """
    Compare model prediction with ground truth mask
    
    Args:
        model_path: Path to the saved model (.pt file)
        image_path: Path to the image to test
        mask_path: Path to the ground truth mask
        output_dir: Directory to save results
        device: Device to run inference on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename for results
    filename_base = Path(image_path).stem
    
    # Load model
    model = load_model(model_path, device)
    
    # Load and preprocess images
    img, gt_mask, img_np, gt_mask_np = load_and_preprocess_images(image_path, mask_path)
    
    # Get model prediction
    seg_output_np, seg_pred, cls_pred, cls_probs = get_model_prediction(model, img, device)
    
    # Print unique values in masks to debug
    print("\nImage Information:")
    print(f"Image: {image_path}")
    print(f"Ground truth classes present: {np.unique(gt_mask_np)}")
    print(f"Predicted classes present: {np.unique(seg_pred)}")
    
    # Print detailed class statistics
    print_class_statistics(seg_pred, gt_mask_np)
    
    # Calculate metrics
    dice_score = dice_coef_metric(seg_pred, gt_mask_np)
    iou_score = iou_metric(seg_pred, gt_mask_np)
    
    print(f"\nDice Score: {dice_score:.4f}")
    print(f"IoU Score: {iou_score:.4f}")
    
    # Create visualization
    output_path = create_combined_visualization(
        img_np, gt_mask_np, seg_pred, cls_pred, cls_probs, 
        dice_score, iou_score, output_dir, filename_base
    )
    
    # Print classification confidence
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    print("\nClassification Confidence:")
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name}: {cls_probs[i]*100:.2f}%")
    
    print(f"\nResult saved to: {output_path}")
    
    # Return metrics for batch processing
    return {
        'filename': filename_base,
        'dice': dice_score,
        'iou': iou_score,
        'predicted_class': cls_pred,
        'true_classes': list(np.unique(gt_mask_np)),
        'class_confidence': cls_probs[cls_pred]
    }

def batch_process(model_path, test_dir, output_dir='batch_results', device='cuda'):
    """
    Process multiple test images and compile results
    
    Args:
        model_path: Path to model
        test_dir: Directory containing test images and masks
        output_dir: Directory to save results
        device: Device to run inference on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    
    # Track results
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        # Find corresponding mask (assuming same name with .png extension)
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(test_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_file}, skipping.")
            continue
        
        print(f"\nProcessing: {img_file}")
        try:
            # Process single image
            result = compare_prediction_with_gt(model_path, img_path, mask_path, output_dir, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Compile summary statistics
    if results:
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        
        print("\nBatch Processing Summary:")
        print(f"Images processed: {len(results)}")
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        
        # Save summary to CSV
        import csv
        csv_path = os.path.join(output_dir, 'results_summary.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'dice', 'iou', 'predicted_class', 'class_confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'filename': r['filename'],
                    'dice': r['dice'],
                    'iou': r['iou'],
                    'predicted_class': r['predicted_class'],
                    'class_confidence': r['class_confidence']
                })
        
        print(f"Results summary saved to: {csv_path}")
    else:
        print("No images were successfully processed.")


if __name__ == "__main__":
    # Path to your trained model
    model_path = r"D:/best_model.pt"
    
    # Path to test image and ground truth mask
    image_path = "test/pitu.jpg"
    mask_path = "test/pitu.png"
    
    # For single image testing
    compare_prediction_with_gt(model_path, image_path, mask_path)
    
    # For batch processing (uncomment to use)
    # batch_process(model_path, "test_dataset")
