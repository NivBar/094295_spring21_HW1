from torch.utils.data import Dataset
import os
import argparse
import pandas as pd
import evaluate
from torchvision import transforms
import torch
import config
import ResNet
import dataset


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is:", device)
torch.cuda.empty_cache()

test_dataset = dataset.costum_images_dataset(root_dir=config.test_root_dir,
                                             transform=transforms.Compose(
                                                 [dataset.Rescale(config.resize_param), dataset.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

models = {
    'clf': ResNet.ResNetClassifier(
        in_size=config.in_size, out_classes=1, channels=[32, 64, 128, 256],
        pool_every=1, hidden_dims=[100] * 2,
        activation_type='relu',
        pooling_type='avg', pooling_params=dict(kernel_size=2),
        batchnorm=True, dropout=0.2, ),

    'reg': ResNet.ResNetClassifier(
        in_size=config.in_size, out_classes=4, channels=[32, 64, 128, 256],
        pool_every=1, hidden_dims=[100] * 2,
        activation_type='relu',
        pooling_type='avg', pooling_params=dict(kernel_size=2),
        batchnorm=True, dropout=0.2, )
    }

clf_model = models['clf']
clf_model.to(device)
tmp_clf_param = torch.load('best_resnet_clf.pkl', map_location=device)
clf_model.load_state_dict(tmp_clf_param)
clf_model.eval()

proper_mask_pred = evaluate.predict_label(clf_model, test_loader, device)

del clf_model
torch.cuda.empty_cache()
#
reg_model = models['reg']
reg_model.to(device)
tmp_reg_param = torch.load('best_resnet_reg.pkl', map_location=device)
reg_model.load_state_dict(tmp_reg_param)
reg_model.eval()

bbox_pred = evaluate.predict_bbox(reg_model, test_loader, device)

prediction_df = pd.DataFrame(zip(files, *bbox_pred, proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
####

prediction_df.to_csv("prediction.csv", index=False, header=True)