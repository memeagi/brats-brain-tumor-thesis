from src.segmentation_model import UNet
from src.classification_model import CNNClassifier

def main():
    segmentation_model = UNet()
    classification_model = CNNClassifier()

    print("Brain Tumor Segmentation and Classification Project")
    print("Segmentation model initialized:", segmentation_model.__class__.__name__)
    print("Classification model initialized:", classification_model.__class__.__name__)

if __name__ == "__main__":
    main()