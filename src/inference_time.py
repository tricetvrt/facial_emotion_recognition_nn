import time
import torch
from model import get_resnet50, get_mobilenetv4, get_mobilevitv2  , get_convnext 

def izmeri_inference_vreme(model, input_size, device, n_warmup=20, n_runs=200, batch_size=1):
    model.eval()
    model.to(device)

    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize() 

        vremena = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()  #ceka da se forward pass STVARNO zavrsi na gpu
            end = time.perf_counter()

            vremena.append((end - start) * 1000)  #u milisekundama

    vremena = torch.tensor(vremena)
    return {
        "mean_ms": vremena.mean().item(),
        "std_ms": vremena.std().item(),
        "min_ms": vremena.min().item(),
        "max_ms": vremena.max().item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uredjaj: {device}\n")

    konfiguracije = [
        ("ResNet50", get_resnet50(num_classes=6, pretrained=False, freeze_backbone=False), 224),
        ("MobileNetV4", get_mobilenetv4(num_classes=6, pretrained=False, freeze_backbone=False), 224),
        ("MobileViTv2", get_mobilevitv2(num_classes=6, pretrained=False, freeze_backbone=False), 256),
        ("ConvNeXt", get_convnext(num_classes=6, pretrained=False, freeze_backbone=False), 224),
    ]

    for naziv, model, input_size in konfiguracije:
        rezultat = izmeri_inference_vreme(model, input_size, device)
        print(f"{naziv:15s} (ulaz {input_size}x{input_size}): "
              f"{rezultat['mean_ms']:.3f} ± {rezultat['std_ms']:.3f} ms "
              f"(min {rezultat['min_ms']:.3f}, max {rezultat['max_ms']:.3f})")


if __name__ == "__main__":
    main()