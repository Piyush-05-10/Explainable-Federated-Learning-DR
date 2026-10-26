import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes=2):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze feature extractor layers to preserve pretrained ImageNet features
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last feature block (block 8: 1x1 conv + BN) to adapt to retina fundus images
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Replace the final linear layer for binary classification (0 = No DR, 1 = DR)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {:,}".format(total_params))
    print("Trainable parameters: {:,}".format(trainable_params))
