import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
import copy
import time
import warnings


torch.backends.cudnn.benchmark = True


DATA_DIR = '../PlantVillage'


IMAGE_SIZE = 128


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


BATCH_SIZE = 32
NUM_EPOCHS = 10 
LEARNING_RATE = 0.001
MOMENTUM = 0.9
PROGRESS_REPORT_INTERVAL = 5 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None, full_dataset_info=None):
        self.subset = subset
        self.transform = transform
        if full_dataset_info:
            self.classes = full_dataset_info.classes
            self.class_to_idx = full_dataset_info.class_to_idx
        else:
            self.classes = getattr(subset.dataset, 'classes', None)
            self.class_to_idx = getattr(subset.dataset, 'class_to_idx', None)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def initialize_model(num_classes, feature_extracting=True, use_pretrained=True):
    model_ft = None
    input_size = IMAGE_SIZE 
    if use_pretrained:

        model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  
        # model_ft = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)

        # model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    else:
        # If not using pretrained weights, initialize from scratch (e.g., ResNet-18)
        model_ft = models.resnet18(weights=None)

    if feature_extracting:
        for param in model_ft.parameters():
            param.requires_grad = False


    if isinstance(model_ft, models.ResNet):
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model_ft, models.MobileNetV2):
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model_ft, models.EfficientNet):
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model type for modifying final layer. Please add logic for your model.")

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1 
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            last_report_time = time.time() 

            num_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

      
                current_time = time.time()
                if current_time - last_report_time >= PROGRESS_REPORT_INTERVAL:
                    current_loss = running_loss / ((batch_idx + 1) * BATCH_SIZE)
                    current_acc = running_corrects.double() / ((batch_idx + 1) * BATCH_SIZE)
                    time_since_start = int(current_time - since)
                    print(f"    {phase} Batch {batch_idx+1}/{num_batches} - Loss: {current_loss:.4f} Acc: {current_acc:.4f} ({time_since_start}s elapsed)")
                    last_report_time = current_time

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'Overall {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            
            if phase == 'val':
             
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'plant_disease_detector_best_model_epoch_{epoch}.pth')
                    print(f"    Saved BEST model checkpoint at epoch {epoch} with Val Acc: {best_acc:.4f}")

         
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': history['train_loss'][-1],
                    'train_acc': history['train_acc'][-1] if len(history['train_acc']) > 0 else 0,
                    'val_loss': epoch_loss,
                    'val_acc': epoch_acc,
                    'best_acc': best_acc,
                    'best_epoch': best_epoch,
                    'class_names': dataloaders['train'].dataset.classes if hasattr(dataloaders['train'].dataset, 'classes') else None # Save class names
                }, f'plant_disease_detector_checkpoint_epoch_{epoch}.pth')
                print(f"    Saved epoch {epoch} training checkpoint.")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} (from epoch {best_epoch})')

    model.load_state_dict(best_model_wts) 
    return model, history


def visualize_model_predictions(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images // 2, 2, images_so_far, xticks=[], yticks=[])
                ax.axis('off')

                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                ax.imshow(img)
                ax.set_title(f'predicted: {class_names[preds[j]]}\nactual: {class_names[labels[j]]}',
                             color=("green" if preds[j]==labels[j] else "red"))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



if __name__ == '__main__':
    print(f"Using device: {device}")


    print("--- Data Preparation ---")
    full_dataset_obj = None
    if os.path.exists(os.path.join(DATA_DIR, 'train')) and os.path.exists(os.path.join(DATA_DIR, 'val')):
        print("Dataset found with 'train' and 'val' subdirectories.")
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                          for x in ['train', 'val']}
        train_dataset = image_datasets['train']
        val_dataset = image_datasets['val']

        class_names = train_dataset.classes
        num_classes = len(class_names)
    else:
        print("No 'train'/'val' subdirectories found. Attempting to split the main dataset.")
        print(f"Loading full dataset from: {DATA_DIR}")
        try:
            full_dataset_obj = datasets.ImageFolder(DATA_DIR)
            print(f"Found {len(full_dataset_obj)} images in total.")

            class_names = full_dataset_obj.classes
            num_classes = len(class_names)

            train_size = int(0.8 * len(full_dataset_obj))
            val_size = len(full_dataset_obj) - train_size
            train_subset, val_subset = random_split(full_dataset_obj, [train_size, val_size])

            train_dataset = TransformedSubset(train_subset, transform=data_transforms['train'], full_dataset_info=full_dataset_obj)
            val_dataset = TransformedSubset(val_subset, transform=data_transforms['val'], full_dataset_info=full_dataset_obj)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure your DATA_DIR is correct and contains class subdirectories directly,")
            print("or has 'train' and 'val' subdirectories.")
            exit()

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print(f"Number of training images: {dataset_sizes['train']}")
    print(f"Number of validation images: {dataset_sizes['val']}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0), 
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }


    print("\n--- Model Initialization ---")
    model, input_size = initialize_model(num_classes, feature_extracting=True, use_pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    print("\n--- Starting Training ---")
    model_ft, history = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=NUM_EPOCHS)
    print("\n--- Training Finished ---")


    print("\n--- Visualizing Predictions ---")
    visualize_model_predictions(model_ft, dataloaders, class_names)
    plt.tight_layout()
    plt.show()

    print("\n--- Plotting Training History ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    print("\nScript finished. Best model and epoch checkpoints saved during training.")