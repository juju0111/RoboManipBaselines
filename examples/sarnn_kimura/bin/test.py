#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
sys.path.append("../../third_party/eipl/")
from tqdm import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization
from model.SARNNwithSideimageAndWrench import SARNNwithSideimageAndWrench

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None, help=".pth file that PyTorch loads as checkpoint for model")
parser.add_argument("--dirname", type=str, default="../simulator/data", help="directory that stores test data, that has been generated, and will be loaded")
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = args.idx

# load dataset
minmax = [params["vmin"], params["vmax"]]
front_images_raw = np.load(os.path.join(args.dirname, "test/front_images.npy"))
side_images_raw = np.load(os.path.join(args.dirname, "test/side_images.npy"))
joints_raw = np.load(os.path.join(args.dirname, "test/joints.npy"))
joint_bounds = np.load(os.path.join(args.dirname, "joint_bounds.npy"))
wrenches_raw = np.load(os.path.join(args.dirname, "test/wrenches.npy"))
wrench_bounds = np.load(os.path.join(args.dirname, "wrench_bounds.npy"))
front_images = front_images_raw[idx]
side_images = side_images_raw[idx]
joints = joints_raw[idx]
wrenches = wrenches_raw[idx]

# define model
model = SARNNwithSideimageAndWrench(
    rec_dim=params["rec_dim"],
    joint_dim=7,
    wrench_dim=6,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[64, 64],
)

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
im_size = 64
front_image_list, side_image_list, joint_list, wrench_list = [], [], [], []
enc_front_pts_list, enc_side_pts_list, dec_front_pts_list, dec_side_pts_list = [], [], [], []
state = None
nloop = len(front_images)
for loop_ct in range(nloop):
    # load data and normalization
    front_img_t = front_images[loop_ct].transpose(2, 0, 1)
    front_img_t = normalization(front_img_t, (0, 255), minmax)
    front_img_t = torch.Tensor(np.expand_dims(front_img_t, 0))
    side_img_t = side_images[loop_ct].transpose(2, 0, 1)
    side_img_t = normalization(side_img_t, (0, 255), minmax)
    side_img_t = torch.Tensor(np.expand_dims(side_img_t, 0))
    joint_t = normalization(joints[loop_ct], joint_bounds, minmax)
    joint_t = torch.Tensor(np.expand_dims(joint_t, 0))
    wrench_t = normalization(wrenches[loop_ct], wrench_bounds, minmax)
    wrench_t = torch.Tensor(np.expand_dims(wrench_t, 0))

    # predict rnn
    y_front_image, y_side_image, y_joint, y_wrench, enc_front_pts, enc_side_pts, dec_front_pts, dec_side_pts, state = model(front_img_t, side_img_t, joint_t, wrench_t, state)

    # denormalization
    pred_front_image = tensor2numpy(y_front_image[0])
    pred_front_image = deprocess_img(pred_front_image, params["vmin"], params["vmax"])
    pred_front_image = pred_front_image.transpose(1, 2, 0)
    pred_side_image = tensor2numpy(y_side_image[0])
    pred_side_image = deprocess_img(pred_side_image, params["vmin"], params["vmax"])
    pred_side_image = pred_side_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)
    pred_wrench = tensor2numpy(y_wrench[0])
    pred_wrench = normalization(pred_wrench, minmax, wrench_bounds)

    # append data
    front_image_list.append(pred_front_image)
    side_image_list.append(pred_side_image)
    joint_list.append(pred_joint)
    wrench_list.append(pred_wrench)
    enc_front_pts_list.append(tensor2numpy(enc_front_pts[0]))
    enc_side_pts_list.append(tensor2numpy(enc_side_pts[0]))
    dec_front_pts_list.append(tensor2numpy(dec_front_pts[0]))
    dec_side_pts_list.append(tensor2numpy(dec_side_pts[0]))

    print("loop_ct:{}, joint:{}, wrench:{}".format(loop_ct, pred_joint, pred_wrench))

pred_front_image = np.array(front_image_list)
pred_side_image = np.array(side_image_list)
pred_joint = np.array(joint_list)
pred_wrench = np.array(wrench_list)

# split key points
enc_front_pts = np.array(enc_front_pts_list)
enc_side_pts = np.array(enc_side_pts_list)
dec_front_pts = np.array(dec_front_pts_list)
dec_side_pts = np.array(dec_side_pts_list)
enc_front_pts = enc_front_pts.reshape(-1, params["k_dim"], 2) * im_size
enc_side_pts = enc_side_pts.reshape(-1, params["k_dim"], 2) * im_size
dec_front_pts = dec_front_pts.reshape(-1, params["k_dim"], 2) * im_size
dec_side_pts = dec_side_pts.reshape(-1, params["k_dim"], 2) * im_size
enc_front_pts = np.clip(enc_front_pts, 0, im_size)
enc_side_pts = np.clip(enc_side_pts, 0, im_size)
dec_front_pts = np.clip(dec_front_pts, 0, im_size)
dec_side_pts = np.clip(dec_side_pts, 0, im_size)

# plot images
T = len(front_images)
fig, ax = plt.subplots(2, 3, figsize=(14, 6), dpi=60)
pbar = tqdm(total=pred_joint.shape[0] + 1, desc=anim.FuncAnimation.__name__)


def anim_update(i):
    for j in range(2):
        for k in range(3):
            ax[j, k].cla()

    # plot camera front_image
    ax[0, 0].imshow(front_images[i])
    for j in range(params["k_dim"]):
        ax[0, 0].plot(enc_front_pts[i, j, 0], enc_front_pts[i, j, 1], "co", markersize=12)  # encoder
        ax[0, 0].plot(
            dec_front_pts[i, j, 0], dec_front_pts[i, j, 1], "rx", markersize=12, markeredgewidth=2
        )  # decoder
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Input front_image", fontsize=20)

    # plot predicted front_image
    ax[0, 1].imshow(pred_front_image[i])
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Predicted front_image", fontsize=20)

    # plot camera side_image
    ax[1, 0].imshow(side_images[i])
    for j in range(params["k_dim"]):
        ax[1, 0].plot(enc_front_pts[i, j, 0], enc_front_pts[i, j, 1], "co", markersize=12)  # encoder
        ax[1, 0].plot(
            dec_front_pts[i, j, 0], dec_front_pts[i, j, 1], "rx", markersize=12, markeredgewidth=2
        )  # decoder
    ax[1, 0].axis("off")
    ax[1, 0].set_title("Input side_image", fontsize=20)

    # plot predicted side_image
    ax[1, 1].imshow(pred_side_image[i])
    ax[1, 1].axis("off")
    ax[1, 1].set_title("Predicted side_image", fontsize=20)

    # plot joint
    ax[0, 2].set_ylim(-np.pi, np.pi * 1.25)
    ax[0, 2].set_yticks([-np.pi / 2, 0, np.pi / 2])
    ax[0, 2].set_xlim(0, T)
    ax[0, 2].plot(joints[1:], linestyle="dashed", c="k")
    for joint_idx in range(pred_joint.shape[1]):
        ax[0, 2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[0, 2].set_xlabel("Step", fontsize=20)
    ax[0, 2].set_title("Joint", fontsize=20)
    ax[0, 2].tick_params(axis="x", labelsize=16)
    ax[0, 2].tick_params(axis="y", labelsize=16)

    # plot wrench
    ax[1, 2].set_ylim(-30, 10)
    ax[1, 2].set_xlim(0, T)
    ax[1, 2].plot(wrenches[1:], linestyle="dashed", c="k")
    for wrench_idx in range(pred_wrench.shape[1]):
        ax[1, 2].plot(np.arange(i + 1), pred_wrench[: i + 1, wrench_idx])
    ax[1, 2].set_xlabel("Step", fontsize=20)
    ax[1, 2].set_title("Wrench", fontsize=20)
    ax[1, 2].tick_params(axis="x", labelsize=16)
    ax[1, 2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9, hspace=0.6)

    pbar.update(1)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 1)), frames=T)
save_file_name = "./output/SARNN_{}_{}.gif".format(params["tag"], idx)
ani.save(save_file_name)
pbar.close()
print("save file: ", save_file_name)

# If an error occurs in generating the gif animation, change the writer (front_imagemagick/ffmpeg).
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")
