from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch
import config
import ResNet
import train
import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is:", device)

train_dataset = dataset.costum_images_dataset(root_dir=config.train_root_dir,
                                              transform=transforms.Compose(
                                                  [dataset.Rescale(config.resize_param), dataset.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

test_dataset = dataset.costum_images_dataset(root_dir=config.test_root_dir,
                                             transform=transforms.Compose(
                                                 [dataset.Rescale(config.resize_param), dataset.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

models = {
    'resnet': ResNet.ResNetClassifier(
        in_size=config.in_size, out_classes=4, channels=[32, 64, 128, 256, 512, 1024],
        pool_every=1, hidden_dims=[100] * 2,
        activation_type='relu',
        pooling_type='avg', pooling_params=dict(kernel_size=2),
        batchnorm=True, dropout=0.2, ),

    'cnn': ResNet.ConvClassifier(
        in_size=config.in_size, out_classes=4, channels=[32, 64] * 2,
        pool_every=2, hidden_dims=[100] * 2,
        activation_type='relu',
        pooling_type='avg', pooling_params=dict(kernel_size=2)
    )}

arc_name = 'resnet'
model_reg_ = models[arc_name]

model_reg_.to(device)
print(model_reg_)
criterion = nn.SmoothL1Loss()
print('number of parameters: ', sum(param.numel() for param in model_reg_.parameters()))
print(f'Num of trainable parameters : {sum(p.numel() for p in model_reg_.parameters() if p.requires_grad)}')
optimizer = torch.optim.Adam(model_reg_.parameters(), lr=config.lr, weight_decay=config.weight_decay)
train.train_model_reg(model_reg_, criterion, optimizer, train_loader, test_loader, device,arc_name)