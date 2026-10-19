import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from src.models.localization import build_localization_model

def run_prediction(image_path, model_path="artifacts/localization_best.pth", output_path="artifacts/prediction_sample.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    model = build_localization_model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load and preprocess the satellite image
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = raw_img.resize((512, 512), Image.Resampling.BILINEAR)
    input_tensor = TF.to_tensor(resized_img).unsqueeze(0).to(device)
    
    # 3. Model forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        # Apply sigmoid to convert raw logits to probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Threshold at 0.5 (above 50% confidence = building)
    binary_pred = (probs > 0.5).astype(np.uint8)
    
    # 4. Save visual side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(resized_img)
    axes[0].set_title("Input Satellite Image (Pre-Disaster)")
    axes[0].axis("off")
    
    axes[1].imshow(resized_img)
    axes[1].imshow(binary_pred, cmap="jet", alpha=0.45)
    axes[1].set_title("ResQVision Predicted Footprints")
    axes[1].axis("off")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"Prediction complete. Saved visual result to: {output_path}")

if __name__ == "__main__":
    # Pick any pre-disaster image from your extracted dataset
    images_dir = r"E:\train\images"
    sample_images = [f for f in os.listdir(images_dir) if f.endswith("_pre_disaster.png")]
    
    if sample_images:
        sample_path = os.path.join(images_dir, sample_images[0])
        run_prediction(sample_path)
    else:
        print("No sample images found in E:\\train\\images")