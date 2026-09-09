import argparse

import train
import evaluate
import dataset


def main():
    parser = argparse.ArgumentParser(
        description="FER+ emotion recognition pipeline"
    )

    parser.add_argument(
        "command",
        choices=["check-data", "train", "evaluate", "train-evaluate"],
        help="komanda koju zelis da pokrenes"
    )

    args = parser.parse_args()

    if args.command == "check-data":
        train_loader, val_loader, test_loader = dataset.get_dataloaders()

        print("Train samples:", len(train_loader.dataset))
        print("Validation samples:", len(val_loader.dataset))
        print("Test samples:", len(test_loader.dataset))
        print("Classes:", dataset.get_class_names())
        print("Class to index:", dataset.get_class_to_idx())

        images, labels = next(iter(train_loader))
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)

    elif args.command == "train":
        train.main()

    elif args.command == "evaluate":
        evaluate.main()

    elif args.command == "train-evaluate":
        train.main()
        evaluate.main()


if __name__ == "__main__":
    main()