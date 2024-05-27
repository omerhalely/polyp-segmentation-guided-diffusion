
# credits: MedSegDiff

import sys

sys.path.append(".")

import numpy as np
import torch
from torch.autograd import Function
import argparse
from polyp_dataset import polyp_dataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.reshape(-1), target.reshape(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred, true_mask_p, threshold=(0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()

            disc_pred = vpred_cpu.numpy().astype('int32')

            disc_mask = gt_vmask_p.squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred, disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred_cpu, gt_vmask_p.cpu()).item()

        return eiou / len(threshold), edice / len(threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT_S8", choices=["DiT_XL2", "DiT_XL4", "DiT_XL8",
                                                                        "DiT_L2", "DiT_L4", "DiT_L8",
                                                                        "DiT_B2", "DiT_B4", "DiT_B8",
                                                                        "DiT_S2", "DiT_S4", "DiT_S8"])
    parser.add_argument("--data-path", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--load-pretrained", type=bool, default=False)
    parser.add_argument("--cross-model", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--num-augmentations", type=int, default=8)
    parser.add_argument("--ema", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--num-testing-steps", type=int, default=100)


    args = parser.parse_args()
    args.cross_model = True if args.cross_model == "true" else False
    args.ema = True if args.ema == "true" else False
    model_name = f"{args.model}_{'CROSS' if args.cross_model else ''}_{args.data_path.split('/')[-1]}_{args.epochs}_epochs_{args.batch_size}_batch_{args.num_augmentations}_augmentations"
    predictions_path = os.path.join(os.getcwd(), "saved_models", model_name, "pred")

    if args.ema:
        predictions_path = predictions_path + f"-ema-{args.num_testing_steps}_testing_steps"
    else:
        predictions_path = predictions_path + f"-{args.num_testing_steps}_testing_steps"
    args.cross_model = True if args.cross_model == "true" else False
    mix_res = (0, 0)
    num = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_dataset = polyp_dataset(
        data_path=args.data_path,
        mode="test",
        device=device
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    data_iter = iter(test_dataloader)

    for i in tqdm(range(len(test_dataloader))):
        gt, image = next(data_iter)
        num += 1
        curr_prediction_path = os.path.join(predictions_path, f"pred_{i + 1}.png")
        pred = plt.imread(curr_prediction_path)
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        gt = gt / 0.18125
        gt = vae.decode(gt).sample

        # make pred dimensions same as gt to fit eval_seg
        pred = torch.from_numpy(np.copy(pred)).to(device)
        pred = torch.unsqueeze(pred, dim=0).to(device)
        pred = torch.cat((pred, pred, pred), dim=0).to(device)
        pred = torch.unsqueeze(pred, dim=0).to(device)

        temp = eval_seg(pred, gt)
        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    iou, dice = tuple([a / num for a in mix_res])
    print('iou is', iou)
    print('dice is', dice)
    #     write a txt
    if args.ema:
        path = os.path.join(os.getcwd(), "saved_models", model_name, f"ema_results_{args.num_testing_steps}_testing_steps.txt")
    else:
        path = os.path.join(os.getcwd(), "saved_models", model_name, f"results_{args.num_testing_steps}_testing_steps.txt")
    with open(path, "w") as f:
        f.write(f"iou is {iou}\n")
        f.write(f"dice is {dice}\n")
    f.close()


if __name__ == "__main__":
    main()