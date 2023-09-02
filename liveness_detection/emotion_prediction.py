import torch
from torch import nn 
from torch.nn import functional as F
import os
from torchvision import transforms as T
import numpy as np
from PIL import Image

class EmotionDetectionModel(nn.Module):
    "VGG-Face"
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(2048, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, 3)
        
    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (batch, 1, 64, 64)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.3, self.training)
        return self.fc8(x)
    
class EmotionPredictor():
    
    def __init__(self, pretrained = 'landmarks/emotion_weights.pt', device = 'cpu', img_size = (64,64), classes = ['smile','surprise', 'neutral']):
        
        if isinstance(device, str):
            if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
                device = torch.device(device)
            else:
                device = torch.device('cpu')
        self.device = device
        
        self.model = EmotionDetectionModel().to(device)
        self.model.eval() 
        
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), pretrained)
            self.model.load_state_dict(torch.load(state_dict_path, map_location= 'cpu'))
            # print('Weights loaded successfully from path:', state_dict_path)
            # print('====================================================')
        
        self.img_size = img_size
        self.classes = np.array(classes) 
    
    def transform(self, image: Image.Image):
        return T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean = [0.5], std = [0.5])
            ]
        )(image) 
        
    def predict(self, image: (np.ndarray, Image.Image, torch.Tensor)):
        """
        Predict the emotion from an input image using the trained model.

        Parameters:
            image (np.ndarray or Image.Image or torch.Tensor): The input image in RGB format.

        Returns:
            str: Predicted emotion class label.

        """

        if isinstance(image, torch.Tensor):
            image = image.numpy()
                
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.convert('L')
        
        tranformed_image = self.transform(image)
        if len(tranformed_image.shape) == 3:
            tranformed_image = tranformed_image[None, ...]

        tranformed_image = tranformed_image.to(self.device)
        out = self.model(tranformed_image)
        
        out = torch.argmax(out).detach().cpu().numpy()
        
        emotion = self.classes[out]
        
        return emotion        

if __name__ == '__main__':
    model = EmotionPredictor()    
    # print(model.img_size)
    
    