import torch
import torch.nn as nn
import torch.optim as optim 
import timm
import seaborn as sns
import wandb
import random
import os
from torch.utils.data import Dataset, DataLoader, Subset  
from torchvision import transforms 
from PIL import Image  
from tqdm import tqdm 

# Enhanced Dataset Class with Error Handling
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        
        # Load images with validation
        self.real_images = self._load_valid_images(real_dir)
        self.fake_images = self._load_valid_images(fake_dir)
        self.images = self.real_images + self.fake_images
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)
        
    def _load_valid_images(self, directory):
        valid_files = []
        for img in os.listdir(directory):
            img_path = os.path.join(directory, img)
            try:
                with Image.open(img_path) as im:
                    im.verify()  # Verify image integrity
                valid_files.append(img_path)
            except Exception as e:
                print(f"Removing corrupt image: {img_path} - {str(e)}")
        return valid_files
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                # Ensure tensor conversion
                if not isinstance(image, torch.Tensor):
                    image = transforms.functional.to_tensor(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return zero tensor and invalid label
            return torch.zeros((3, 224, 224)), -1

# Optimized Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Must come before normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lightweight Model Architecture
class SwinDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('swin_large_patch4_window7_224',
                                        pretrained=True,
                                        num_classes=0)
        self.classifier = nn.Sequential(
    nn.Linear(1536, 512),  # Changed from 256
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

        
    def forward(self, x):
        return self.classifier(self.backbone(x))

# Training Parameters (Preserved Paths)
BATCH_SIZE = 8  # Reduced for stability
LEARNING_RATE = 2e-5
NUM_EPOCHS = 12

# Original Paths (Preserved)
train_real_dir = '/data1/siddharth/data/train/Real'
train_fake_dir = '/data1/siddharth/data/train/Fake'
val_real_dir = '/data1/siddharth/data/validation/Real'
val_fake_dir = '/data1/siddharth/data/validation/Fake'
test_real_dir = '/data1/siddharth/data/test/Real'
test_fake_dir = '/data1/siddharth/data/test/Fake'

# Model Save Path (Updated model name)
model_save_dir = '/data1/kuldeep_2/models/swin_new_tiny_patch4_window7_224'
os.makedirs(model_save_dir, exist_ok=True)

def get_balanced_dataset(dataset):
    real_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    fake_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    min_count = min(len(real_indices), len(fake_indices))
    balanced_indices = random.sample(real_indices, min_count) + random.sample(fake_indices, min_count)
    return Subset(dataset, balanced_indices)

# Optimized Data Loaders
def create_data_loaders():
    train_dataset = DeepfakeDataset(train_real_dir, train_fake_dir, transform)
    val_dataset = DeepfakeDataset(val_real_dir, val_fake_dir, transform)
    test_dataset = DeepfakeDataset(test_real_dir, test_fake_dir, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        get_balanced_dataset(val_dataset),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        get_balanced_dataset(test_dataset),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# Initialize Training Components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader = create_data_loaders()
model = SwinDeepfakeDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

# Enhanced Training Loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    wandb.init(project="swin_tiny_patch4_window7_224")
    
    # Enable performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_counter = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        
        for images, labels in train_progress:
            batch_counter += 1
            
            # Skip invalid batches
            if (labels == -1).any():
                print(f"Skipping invalid batch {batch_counter}")
                continue
                
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Progress monitoring
            if batch_counter % 10 == 0:
                mem = torch.cuda.memory_allocated()/1e9
                print(f"Batch {batch_counter} | Loss: {loss.item():.4f} | GPU Mem: {mem:.2f}GB")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                if (labels == -1).any():
                    continue
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Save checkpoint
        
        model_save_path = os.path.join(model_save_dir,f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        
        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total if total > 0 else 0
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")
        
        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc
        })

    print("Training completed successfully!")

# Start training
if __name__ == "__main__":
    wandb.login()
    train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)