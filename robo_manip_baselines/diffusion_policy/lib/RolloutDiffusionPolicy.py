import os
import sys
import argparse
import hydra
import numpy as np
import matplotlib.pylab as plt
import cv2
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from robo_manip_baselines.common.rollout import RolloutBase

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy"))
print("Path : ", os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy") )


class RolloutDiffusionPolicy(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument(
            "--checkpoint",
            type=str,
            help="checkpoint file of diffusion policy (*.ckpt)",
            required=True,
        )

        super().setup_args(parser)

        if self.args.skip is None:
            self.args.skip = 3
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setup_policy(self):
        # Define policy
        ckpt_data = torch.load(self.args.checkpoint)
        cfg = ckpt_data["cfg"]
        self.policy = hydra.utils.instantiate(cfg.policy)

        # Load weight
        print(f"[RolloutDiffusionPolicy] Load {self.args.checkpoint}")
        self.policy.load_state_dict(ckpt_data["state_dicts"]["ema_model"])
        self.policy.cuda()
        self.policy.eval()

        # Set variables
        self.joint_dim = cfg.shape_meta.action.shape[0]
        self.joint_scales = [1.0] * (self.joint_dim - 1) + [0.01]
        self.image_size = tuple(cfg.task.image_shape[1:][::-1])
        self.n_obs_steps = cfg.n_obs_steps
        self.front_image_history = None
        self.obs_joint_history = None
        self.future_action_seq = []
        self.pred_action_list = np.empty((0, self.joint_dim))

    def setup_plot(self):
        fig_ax = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=60, squeeze=False)
        super().setup_plot(fig_ax=fig_ax)

    def infer_policy(self):
        if self.auto_time_idx % self.args.skip != 0:
            return False

        # Set observation history
        self.front_image = cv2.resize(self.info["rgb_images"]["front"], self.image_size)
        if self.front_image_history is None:
            self.front_image_history = []
            for _ in range(self.n_obs_steps):
                self.front_image_history.append(self.front_image.copy())
        else:
            self.front_image_history.pop(0)
            self.front_image_history.append(self.front_image)
        obs_joint = self.motion_manager.get_joint_pos(self.obs)
        if self.obs_joint_history is None:
            self.obs_joint_history = []
            for _ in range(self.n_obs_steps):
                self.obs_joint_history.append(obs_joint.copy())
        else:
            self.obs_joint_history.pop(0)
            self.obs_joint_history.append(obs_joint)

        if len(self.future_action_seq) == 0:
            inference_called = True

            # Preprocess
            front_image_history_input = np.moveaxis(
                np.array(self.front_image_history).astype(np.float32) / 255, -1, 1
            )
            obs_joint_history_input = np.array(self.obs_joint_history).astype(
                np.float32
            )
            obs_dict_input = {
                "image": np.expand_dims(front_image_history_input, 0),
                "joint": np.expand_dims(obs_joint_history_input, 0),
            }
            obs_dict_input = dict_apply(
                obs_dict_input,
                lambda x: torch.from_numpy(x).to(device=self.policy.device),
            )

            # Infer
            action_dict_output = self.policy.predict_action(obs_dict_input)
            action_dict_output = dict_apply(
                action_dict_output, lambda x: x.detach().to("cpu").numpy()
            )
            self.future_action_seq = list(action_dict_output["action"][0])
        else:
            inference_called = False

        # Store predicted action
        self.pred_action = self.future_action_seq.pop(0)
        self.pred_action_list = np.concatenate(
            [self.pred_action_list, np.expand_dims(self.pred_action, 0)]
        )

        return inference_called

    def draw_plot(self):
        if self.auto_time_idx % self.args.skip_draw != 0:
            return

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Draw observed image
        self.ax[0, 0].imshow(self.front_image)
        self.ax[0, 0].set_title("Observed image", fontsize=20)

        # Plot joint
        xlim = 500 // self.args.skip
        self.ax[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax[0, 1].set_xlim(0, xlim)
        for joint_idx in range(self.pred_action_list.shape[1]):
            self.ax[0, 1].plot(
                np.arange(self.pred_action_list.shape[0]),
                self.pred_action_list[:, joint_idx] * self.joint_scales[joint_idx],
            )
        self.ax[0, 1].set_xlabel("Step", fontsize=20)
        self.ax[0, 1].set_title("Joint", fontsize=20)
        self.ax[0, 1].tick_params(axis="x", labelsize=16)
        self.ax[0, 1].tick_params(axis="y", labelsize=16)
        self.ax[0, 1].axis("on")

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
