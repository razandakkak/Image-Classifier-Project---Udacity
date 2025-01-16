import argparse
import json
import torch
from torchvision import models
from PIL import Image
import numpy as np

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name and probability from an image.")

    parser.add_argument('input', type=str, help="Path to the input image.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint.")
    parser.add_argument('--top_k', type=int, default=1, help="Return top K most likely classes.")
    parser.add_argument('--category_names', type=str, help="Path to JSON file mapping categories to real names.")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference.")

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    # Load the model architecture
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture {arch}.")

    # Rebuild the classifier
    input_size = model.classifier[0].in_features if arch == 'vgg16' else model.fc.in_features
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_units, 102),  # Assuming 102 flower classes
        torch.nn.LogSoftmax(dim=1)
    )

    if arch == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    image = Image.open(image_path)

    # Apply transformations: Resize, Crop, Normalize
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))

    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image).float()

def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        ps = torch.exp(outputs)

    top_p, top_class = ps.topk(top_k, dim=1)

    # Reverse the class_to_idx mapping to get the class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_class.cpu().numpy()[0]]

    return top_p.cpu().numpy()[0], top_classes

def load_category_names(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    args = get_input_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.input, model, args.top_k, device)

    if args.category_names:
        category_names = load_category_names(args.category_names)
        class_names = [category_names[str(cls)] for cls in classes]
    else:
        class_names = classes

    print("Predictions:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
