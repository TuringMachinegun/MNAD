import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.data_loader import DataSet
from model.model import ConvAE

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--h", type=int, default=256, help="height of input images")
parser.add_argument("--w", type=int, default=256, help="width of input images")
parser.add_argument("--c", type=int, default=3, help="channel of input images")
parser.add_argument("--task", type=str, default="prediction", help="The target task for anoamly detection")
parser.add_argument("--t_length", type=int, default=5, help="length of the frame sequences")
parser.add_argument("--num_workers", type=int, default=2, help="number of workers for the loader")

parser.add_argument("--epochs", type=int, default=60, help="number of epochs for training")
parser.add_argument("--loss_compact", type=float, default=0.1, help="weight of the feature compactness loss")
parser.add_argument("--loss_separate", type=float, default=0.1, help="weight of the feature separateness loss")
parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
parser.add_argument("--mdim", type=int, default=512, help="channel dimension of the memory items")
parser.add_argument("--msize", type=int, default=10, help="number of the memory items")
parser.add_argument("--exp_dir", type=str, default="log", help="directory of log")
parser.add_argument(
    "--train_path",
    type=str,
    help="Path to train dataset. Each folder in the path corresponds to a video, and it contains frames as .jpg, "
    "in lexicographic order."
)

args = parser.parse_args()

log_dir = args.exp_dir

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
train_dataset = DataSet(
    args.train_path,
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)
train_batch = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)
train_size = len(train_dataset)


# Model setting
assert args.task == "prediction" or args.task == "reconstruction", "Wrong task name"
t_len = args.t_length if args.task == "prediction" else 2
model = ConvAE(
    args.c,
    t_length=t_len,
    task=args.task,
)
params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
model.cuda()

# Training
loss_func_mse = nn.MSELoss(reduction="none")

m_items = F.normalize(
    torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1
).cuda()  # Initialize the memory items

for epoch in range(args.epochs):
    labels_list = []
    model.train()

    start = time.time()
    for j, (imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        if args.task == "prediction":
            imgs_input = imgs[:, 0:12]  # TODO: looks like first 4 frames?
            out_truth = imgs[:, 12:]  # TODO: looks like last frame?
        else:
            imgs_input = imgs
            out_truth = imgs
        (
            outputs,
            _,
            _,
            m_items,
            softmax_score_query,
            softmax_score_memory,
            separateness_loss,
            compactness_loss,
        ) = model.forward(imgs_input, m_items, True)

        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, out_truth))
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()

    print("----------------------------------------")
    print("Epoch:", epoch + 1)
    print(
        "Loss: {} {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}".format(
            args.task.capitalize(),
            loss_pixel.item(),
            compactness_loss.item(),
            separateness_loss.item(),
        )
    )

    print("Memory_items:")
    print(m_items)
    print("----------------------------------------")

print("Training is finished")

# Save the model and the memory items
torch.save(model, os.path.join(log_dir, "model.pt"))
torch.save(m_items, os.path.join(log_dir, "keys.pt"))
