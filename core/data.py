import torch
import glob
import imageio
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_root):
        self.data = glob.glob(f"{data_root}/*.jpg")
        self.transform = transforms.Compose(
            transforms.ToTensor()
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = imageio.imread(self.data[index])
        return self.transform(img)
