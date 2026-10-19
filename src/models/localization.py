import torch
import segmentation_models_pytorch as smp

def build_localization_model(device="cuda"):
    """
    Creates a U-Net model for finding buildings in pre-disaster images.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # The 'brain' extracting features
        encoder_weights="imagenet",     # Pre-trained on standard images for a head start
        in_channels=3,                  # Input is standard RGB images (3 channels)
        classes=1,                      # Output is a single binary mask (1=building, 0=background)
    )
    
    print(f"Localization model successfully built and sent to: {device}")
    return model.to(device)

if __name__ == "__main__":
    # Test to ensure the model builds successfully on your RTX 2050
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_localization_model(device=device)
    
    # Simulate a dummy image tensor (Batch Size 4, 3 Channels, 512x512 Pixels)
    dummy_input = torch.randn(4, 3, 512, 512).to(device)
    
    # Pass it through the model to verify output shape
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} -> (Batch, Classes, Height, Width)")