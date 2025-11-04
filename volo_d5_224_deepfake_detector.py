import torch.nn as nn  
import timm  

class VOLODeepfakeDetector(nn.Module):  
    def __init__(self, num_classes=2):  
        super(VOLODeepfakeDetector, self).__init__()  
        # Use VOLO d5 model 
        self.backbone = timm.create_model('volo_d5_224', pretrained=True, num_classes=0)  
        
        # Classifier head 
        self.classifier = nn.Sequential( 
            nn.Linear(self.backbone.num_features, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(512, num_classes) 
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)