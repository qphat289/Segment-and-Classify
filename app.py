import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from skimage import measure
from src.model import MobileNetUNet  # Adjust the import path as needed

# ---------------------------
# Helper functions
# ---------------------------
def load_model(model_path, device):
    """Load and return the segmentation model."""
    model = MobileNetUNet(img_ch=1, seg_ch=4, num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(image, model, device):
    """Preprocess the image, run prediction, and return data for visualization."""
    # Convert image to grayscale and resize
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((256, 256))
    img_np = np.array(image)

    # Prepare tensor for model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_output, cls_output = model(img_tensor)
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
        cls_pred = torch.argmax(cls_output, dim=1).item()
        cls_probs = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()
    
    return img_np, seg_pred, cls_pred, cls_probs

def plot_result(img_np, seg_pred, cls_pred, cls_probs):
    """Create and return a matplotlib figure with segmentation overlay and contour."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np, cmap='gray')

    # Create a binary mask (non-zero indicates tumor)
    mask = seg_pred > 0
    if np.any(mask):
        # Create a semi-transparent orange overlay
        overlay = np.zeros((*img_np.shape, 4))
        overlay[mask] = [1, 0.5, 0, 0.3]
        ax.imshow(overlay)
        # Find and draw contours
        contours = measure.find_contours(mask.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 5:
                ax.plot(contour[:, 1], contour[:, 0], color='orange', linewidth=2)

    ax.axis('off')
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    ax.set_title(f"{class_names[cls_pred]} (Confidence: {cls_probs[cls_pred]*100:.1f}%)", fontsize=14)
    return fig

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Brain Tumor Segmentation")
    st.write("Upload an image to generate segmentation predictions.")

    # File uploader for input image
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Run Prediction"):
        if image_file is not None:
            # Save uploaded image to a temporary directory
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, image_file.name)
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            image = Image.open(image_path)
            
            # Load model and perform prediction
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = "kaggle/checkpoint_ag/kaggle/working/Segment-and-Classify/checkpoint_new/best_model.pt"  # Adjust path as needed
            model = load_model(model_path, device)
            st.write("Running prediction, please wait...")
            img_np, seg_pred, cls_pred, cls_probs = predict(image, model, device)
            fig = plot_result(img_np, seg_pred, cls_pred, cls_probs)

            # Display input and prediction result side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.header("Input Image")
                st.image(img_np, use_container_width=True)
            with col2:
                st.header("Prediction Result")
                st.pyplot(fig)
                st.write(f"Predicted: {['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'][cls_pred]} (Confidence: {cls_probs[cls_pred]*100:.1f}%)")
            st.success("Prediction completed!")
        else:
            st.error("Please upload an image.")

if __name__ == "__main__":
    main()
