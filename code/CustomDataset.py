from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):

    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        past_img = Image.open(self.img_paths[index])
        past_img = self.transform(past_img)

        future_img = Image.open(self.img_paths[index+1])
        future_img = self.transform(future_img)

        return future_img, past_img

    def __len__(self):

        return len(self.img_paths)-1