import torch
import torch.nn as nn
import timm
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_mobilenetv4(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True,
                     model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"):

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    # kroz model name prosledjujemo verziju modela
    # da li je pretreniran
    # broj klasa
    
 
    if freeze_backbone:
        for name, param in model.named_parameters():
            # classifier je naziv poslednjeg sloja u timm MobileNetV4
            if "classifier" not in name:
                param.requires_grad = False # zamrzavamo svaki sloj sem poslednjeg ne racuna gradijent za ove slojeve
 
    return model

def get_mobilevitv2(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True,
                     model_name: str = "mobilevitv2_100.cvnets_in1k"):

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        # umesto pretpostavke o nazivu poslednjeg sloja (kao kod mobilenetv4 "classifier"),
        # ovde koristimo timm-ov get_classifier() da bismo bili sigurni koji su parametri
        # klasifikacione glave, bez obzira na internu arhitekturu MobileViTv2
        classifier_params = set(model.get_classifier().parameters())
        for param in model.parameters():
            if param not in classifier_params:
                param.requires_grad = False

    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = get_resnet50(num_classes=6, pretrained=True, freeze_backbone=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape RESNET-50:", output.shape)