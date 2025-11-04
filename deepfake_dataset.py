import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.images = self.real_images + self.fake_images
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
