import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations
import cv2


class TrainDataset(Dataset):
    def __init__(self, data_root, size=None):
        self.size = size

        self.images = [os.path.join(data_root, file) for file in os.listdir(data_root)]
        self._length = len(self.images)

        self.rescaler = albumentations.LongestMaxSize(max_size=256)
        self.padding = albumentations.PadIfNeeded(256,256,border_mode= cv2.BORDER_CONSTANT,value=[223,223,223])
        self.preprocessor = albumentations.Compose([self.rescaler, self.padding])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


# dataset = TrainDataset('data/xray/train',256)
# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset,batch_size=1)

# a = next(iter(dataloader))
# print('asd')