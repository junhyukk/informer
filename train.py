# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import utils
import time
import datetime
import os 

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from compressai.datasets import ImageFolder
from compressai.zoo.image import cfgs

from models.informer import Informer


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metric="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["rate_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == "mse":
            out["distortion_loss"] = 255 ** 2 * self.mse(output["x_hat"], target)
        elif self.metric == "ms-ssim":
            out["distortion_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
        out["loss"] = self.lmbda * out["distortion_loss"] + out["rate_loss"]
        return out


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch {} (train):'.format(epoch)    
    print_freq = 10 

    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    for d in metric_logger.log_every(train_dataloader, print_freq, header):
        d = d.to(device, non_blocking=True)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        metric_logger.update(loss=out_criterion["loss"].item())
        metric_logger.update(rate_loss=out_criterion["rate_loss"].item())
        metric_logger.update(distortion_loss=out_criterion["distortion_loss"].item())
        metric_logger.update(aux_loss=aux_loss.item())

    print(header, "Averaged stats:", metric_logger, "\n")
    return {k: meter.global_avg for k, meter, in metric_logger.meters.items()}


def test_forward(epoch, dataset_name, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    metric_logger = utils.MetricLogger(delimiter= "  ")
    header = f'Epoch {epoch} (test, {dataset_name}):'
    print_freq = 10

    with torch.no_grad():
        for d in metric_logger.log_every(test_dataloader, print_freq, header):
            d = d.to(device, non_blocking=True)
            h, w = d.size(2), d.size(3)

            out_net = model(d)

            num_pixels = h * w
            tmp_bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )

            tmp_psnr = psnr(d, out_net["x_hat"])

            metric_logger.update(bpp=tmp_bpp.item())
            metric_logger.update(psnr=tmp_psnr)

    print(header, "Averaged stats:", metric_logger, "\n")
    return {k: meter.global_avg for k, meter, in metric_logger.meters.items()}
    

def test_codec(epoch, dataset_name, test_dataloader, model, metric):
    model.eval()
    model.update(force=True)
    device = next(model.parameters()).device

    metric_logger = utils.MetricLogger(delimiter= "  ")
    header = f'Epoch {epoch} (test, {dataset_name}):'
    print_freq = 100

    with torch.no_grad():
        for d in metric_logger.log_every(test_dataloader, print_freq, header):
            d = d.to(device)
            h, w = d.size(2), d.size(3)

            out_enc = model.compress(d)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

            num_pixels = h * w
            tmp_bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

            metric_logger.update(bpp=tmp_bpp)
            
            if metric == "mse":
                tmp_psnr = psnr(d, out_dec["x_hat"])
                metric_logger.update(psnr=tmp_psnr)
            elif metric == "ms-ssim":
                tmp_ms_ssim = ms_ssim(d, out_dec["x_hat"], data_range=1.0).item()
                metric_logger.update(ms_ssim=tmp_ms_ssim)

    print(header, "Averaged stats:", metric_logger, "\n")
    return {k: meter.global_avg for k, meter, in metric_logger.meters.items()}


def save_checkpoint(state, filename="checkpoint.pth.tar", is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "--test-dataset", type=str, help="Test dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0067,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3,
        help="Network architecture parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3269', help='url used to set up distributed training')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.backends.cudnn.deterministic = True
        seed = args.seed
        torch.manual_seed(seed)
        random.seed(seed)

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)

    tb_logger = SummaryWriter(log_dir=args.experiment+'/log')

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    kodak_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    # tecnick_transforms = transforms.Compose(
    #     [transforms.ToTensor()]
    # )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    kodak_dataset = ImageFolder(args.test_dataset, split="kodak", transform=kodak_transforms)
    # tecnick_dataset = ImageFolder(args.test_dataset, split="tecnick", transform=tecnick_transforms)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        shuffle=True,
    )
    kodak_dataloader = DataLoader(
        kodak_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        shuffle=False,
    )
    # tecnick_dataloader = DataLoader(
    #     tecnick_dataset,
    #     sampler=tecnick_sampler, 
    #     batch_size=1,
    #     num_workers=args.num_workers,
    #     pin_memory=(device == "cuda"),
    #     drop_last=False,
    # )

    net = Informer(*cfgs["mbt2018"][args.quality], num_global=8)
    net = net.to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('\n# params:', n_parameters)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180,210,240], gamma=1/3)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"Start training for {args.epochs} epochs\n")
    start_time = time.time()
    for epoch in range(last_epoch, args.epochs):

        train_stats = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )

        lr_scheduler.step()

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.experiment, "checkpoint.pth.tar")
            )

            if (epoch+1) % 50 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(), 
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.experiment, "checkpoint_{:03d}.pth.tar".format(epoch+1))
                )

        if (epoch+1) % 5 == 0:
            kodak_codec = test_codec(epoch, "kodak", kodak_dataloader, net, args.metric)

            for k, v in train_stats.items():
                tb_logger.add_scalar(f'train_{k}', v, epoch)

            for k, v in kodak_codec.items():
                tb_logger.add_scalar(f'kodak_{k}', v, epoch)

        else:
            for k, v in train_stats.items():
                tb_logger.add_scalar(f'train_{k}', v, epoch)


    sys.stdout.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main(sys.argv[1:])