import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model checkpoint.")

    parser.add_argument('data_dir', type=str, help="Directory containing the dataset.")
    parser.add_argument('--save_dir', type=str, default='./', help="Directory to save the model checkpoint.")
    parser.add_argument('--arch', type=str, default='vgg16', help="Model architecture (e.g., 'vgg16', 'resnet18').")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in the classifier.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training.")

    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return train_loader, valid_loader, train_data

def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture {arch}. Please choose 'vgg16' or 'resnet18'.")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    input_size = model.classifier[0].in_features if arch == 'vgg16' else model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),  # Assuming 102 flower classes
        nn.LogSoftmax(dim=1)
    )

    if arch == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier

    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

def save_checkpoint(model, save_dir, arch, hidden_units, train_data):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    args = get_input_args()

    train_loader, valid_loader, train_data = load_data(args.data_dir)

    model = build_model(arch=args.arch, hidden_units=args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(),
                           lr=args.learning_rate)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, train_data)

if __name__ == "__main__":
    main()
