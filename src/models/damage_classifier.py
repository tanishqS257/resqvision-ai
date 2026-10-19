import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class TwinTowerDamageClassifier(nn.Module):
    """
    Siamese Twin-Tower network for damage classification.
    Processes pre- and post-disaster images with a shared backbone,
    extracts feature representations, and computes element-wise feature
    differences to detect structural destruction.
    """
    def __init__(self, encoder_name="resnet34", num_classes=5):
        # 5 output classes: 0 = background, 1 = no-damage, 2 = minor, 3 = major, 4 = destroyed
        super().__init__()
        
        self.base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        self.encoder = self.base_unet.encoder
        self.decoder = self.base_unet.decoder
        self.segmentation_head = self.base_unet.segmentation_head

    def forward(self, pre_img, post_img):
        # 1. Extract feature maps using shared encoder
        pre_features = self.encoder(pre_img)
        post_features = self.encoder(post_img)
        
        # 2. Element-wise absolute difference at each resolution level
        diff_features = [torch.abs(post - pre) for pre, post in zip(pre_features, post_features)]
        
        # 3. Pass features as a list/tuple to the decoder
        decoder_output = self.decoder(diff_features)
        logits = self.segmentation_head(decoder_output)
        return logits

def build_damage_model(device="cuda"):
    model = TwinTowerDamageClassifier(encoder_name="resnet34", num_classes=5)
    return model.to(device)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_damage_model(device=device)
    
    dummy_pre = torch.randn(2, 3, 512, 512).to(device)
    dummy_post = torch.randn(2, 3, 512, 512).to(device)
    out = model(dummy_pre, dummy_post)
    print(f"Damage Model Output shape: {out.shape} -> (Batch, 5 classes, 512, 512)")