import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import json


class costum_images_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, sorted(os.listdir(self.root_dir))[idx])
        image = Image.open(img_name)
        # .split('/')[-1]
        bbox = img_name.strip(".jpg").split("__")[-2]
        bbox = json.loads(bbox)
        label = img_name.strip(".jpg").split("__")[-1]
        label = True if label.lower() == "true" else False
        # label = img_name[-5]  # get the label of a given image
        sample = (image, bbox, int(label))

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox, label = sample

        h, w = image.height, image.width
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = torchvision.transforms.Resize((new_h, new_w))(image)

        return (img, bbox, label)


class ToTensor(object):
    def __call__(self, sample):
        image, bbox, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image)
        image = image.transpose((2, 0, 1)).astype(float)
        img_to_tensor = torch.from_numpy(image)
        return img_to_tensor, torch.tensor(bbox), torch.tensor(label)