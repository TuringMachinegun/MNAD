import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.data_loader import DataSet
from utils import (
    point_score,
    score_sum,
    anomaly_score_list,
    anomaly_score_list_inv,
    AUC,
    psnr,
)

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--h", type=int, default=256, help="height of input images")
parser.add_argument("--w", type=int, default=256, help="width of input images")
parser.add_argument("--c", type=int, default=3, help="channel of input images")
parser.add_argument("--task", type=str, default="prediction", help="The target task for anomaly detection")
parser.add_argument("--t_length", type=int, default=5, help="length of the frame sequences")
parser.add_argument("--num_workers", type=int, default=1, help="number of workers for the loader")

parser.add_argument("--alpha", type=float, default=0.6, help="weight for the anomality score")
parser.add_argument("--th", type=float, default=0.01, help="threshold for test updating")
parser.add_argument(
    "--test_path",
    type=str,
    help="Path to test dataset. Each folder in the path corresponds to a video, and it contains frames as .jpg, "
    "in lexicographic order, and a labels.npy file "
    "(saved numpy int8 array of same length as number of frames in video, 0=normal, 1=anomaly)",
)
parser.add_argument("--model_dir", type=str, help="directory of model")
parser.add_argument("--m_items_dir", type=str, help="directory of model")

args = parser.parse_args()
assert args.task == "prediction" or args.task == "reconstruction", "Wrong task name"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

# Loading dataset
test_dataset = DataSet(
    args.test_path,
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)

test_size = len(test_dataset)

test_batch = data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False,
)

loss_func_mse = nn.MSELoss(reduction="none")

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
m_items_test = m_items.clone()

model.eval()

psnr_lists = defaultdict(list)
feature_distance_lists = defaultdict(list)

for k, (imgs) in enumerate(test_batch):
    imgs = Variable(imgs).cuda()

    if args.task == "prediction":
        imgs_input = imgs[:, 0:12]  # TODO: looks like first 4 frames?
        out_truth = imgs[:, 12:]  # TODO: looks like last frame?
    else:
        imgs_input = imgs
        out_truth = imgs

    (
        outputs,
        feas,
        updated_feas,
        m_items_test,
        softmax_score_query,
        softmax_score_memory,
        compactness_loss,
        _,
    ) = model.forward(imgs_input, m_items_test, False)
    mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (out_truth[0] + 1) / 2)).item()
    mse_feas = compactness_loss.item()

    # Calculating the threshold for updating at the test time
    point_sc = point_score(outputs, out_truth)

    if point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test)

    psnr_lists[test_dataset.seq_idx_originating_video[k]].append(psnr(mse_imgs))
    feature_distance_lists[test_dataset.seq_idx_originating_video[k]].append(mse_feas)


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video_name in test_dataset.videos:
    anomaly_score_total_list += score_sum(
        anomaly_score_list(psnr_lists[video_name]),
        anomaly_score_list_inv(feature_distance_lists[video_name]),
        args.alpha,
    )

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - test_dataset.get_labels(), 0))

print("AUC: ", accuracy * 100, "%")
