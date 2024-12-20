import torch
import numpy as np
import sys
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from robo_manip_baselines.act import load_data

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy
from detr.models.detr_vae import DETRVAE

# import detr.models.detr_vae

# print(detr.models.detr_vae.DETRVAE.__module__)
# print(detr.models.detr_vae.__file__)
# print("Detr-vae : ", DETRVAE)

class TrainAct(object):
    def __init__(self, jupyter=False, **kwargs):
        if not jupyter:
            self.setup_args()
        else:
            print("kwargs : ", kwargs)
            self.setup_args_jupyter(kwargs)

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self):
        parser = argparse.ArgumentParser(description="Train ACT")

        parser.add_argument(
            "--dataset_dir",
            default="./data/",
            type=str,
            help="dataset_dir",
        )
        parser.add_argument(
            "--log_dir",
            default="./log/",
            type=str,
            help="log_dir",
        )
        parser.add_argument(
            "--camera_names",
            action="store",
            type=lambda x: list(map(str, x.split(","))),
            help="camera_names",
            required=False,
            default=["front"],
        )
        parser.add_argument("--batch_size", default=8, type=int, help="batch_size")
        parser.add_argument("--seed", default=0, type=int, help="seed")
        parser.add_argument("--num_epochs", default=1000, type=int, help="num_epochs")
        parser.add_argument("--lr", default=1e-5, type=float, help="lr")

        # for ACT
        parser.add_argument("--kl_weight", default=10, type=int, help="KL weight")
        parser.add_argument(
            "--chunk_size", default=100, type=int, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", default=512, type=int, help="hidden dimension of ACT policy"
        )
        parser.add_argument(
            "--dim_feedforward",
            default=3200,
            type=int,
            help="feedforward dimension of ACT policy",
        )
        parser.add_argument('--ckpt_dir', action='store', type=str, default="./output", help='ckpt_dir')
        parser.add_argument('--policy_class', action='store', type=str, default="policy_class", help='policy_class, capitalize')
        parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
        parser.add_argument('--state_dim', action='store', type=int, help='state_dim', required=True)
        parser.add_argument('--use_jupyter', action='store', type=bool, help='state_dim', default=False)

        self.args = parser.parse_args()


    def setup_args_jupyter(self, kwargs:dict):
        parser = argparse.ArgumentParser(description="Train ACT")

        parser.add_argument(
            "--dataset_dir",
            default="./data/",
            type=str,
            help="dataset_dir",
        )
        parser.add_argument(
            "--log_dir",
            default="./log/",
            type=str,
            help="log_dir",
        )
        parser.add_argument(
            "--camera_names",
            action="store",
            type=lambda x: list(map(str, x.split(","))),
            help="camera_names",
            required=False,
            default=["front"],
        )
        parser.add_argument("--batch_size", default=8, type=int, help="batch_size")
        parser.add_argument("--seed", default=0, type=int, help="seed")
        parser.add_argument("--num_epochs", default=1000, type=int, help="num_epochs")
        parser.add_argument("--lr", default=1e-5, type=float, help="lr")

        # for ACT
        parser.add_argument("--kl_weight", default=10, type=int, help="KL weight")
        parser.add_argument(
            "--chunk_size", default=100, type=int, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", default=512, type=int, help="hidden dimension of ACT policy"
        )
        parser.add_argument(
            "--dim_feedforward",
            default=3200,
            type=int,
            help="feedforward dimension of ACT policy",
        )
        parser.add_argument('--ckpt_dir', action='store', type=str, default="./output", help='ckpt_dir')
        parser.add_argument('--policy_class', action='store', type=str, default="policy_class", help='policy_class, capitalize')
        parser.add_argument('--task_name', action='store', type=str, help='task_name', default="")
        parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=7)
        parser.add_argument('--use_jupyter', action='store', type=bool, help='state_dim', default=True)

        self.args = parser.parse_args([])

        # kwargs를 self.args에 동적으로 업데이트
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                print(f"Updating {key} to {value}")
            else:
                print(f"Adding new argument {key} with value {value}")
            setattr(self.args, key, value)
        
    def setup_dataset(self):
        set_seed(1)

        is_sim = True
        batch_size_train = self.args.batch_size
        batch_size_val = self.args.batch_size
        self.train_dataloader, self.val_dataloader, self.stats, _ = load_data(
            self.args.dataset_dir,
            is_sim,
            self.args.camera_names,
            batch_size_train,
            batch_size_val,
        )

    def setup_policy(self):
        set_seed(self.args.seed)

        state_dim = self.train_dataloader.dataset[0][1].shape[0]
        action_dim = self.train_dataloader.dataset[0][2].shape[1]

        print("state dim : ", state_dim,"action dim : ",action_dim)

        # DETRVAE.set_state_dim(state_dim)
        # DETRVAE.set_action_dim(action_dim)

        lr_backbone = 1e-5
        backbone = "resnet18"
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr"                : self.args.lr,
            "num_queries"       : self.args.chunk_size,
            "kl_weight"         : self.args.kl_weight,
            "hidden_dim"        : self.args.hidden_dim,
            "dim_feedforward"   : self.args.dim_feedforward,
            "lr_backbone"       : lr_backbone,
            "backbone"          : backbone,
            "enc_layers"        : enc_layers,
            "dec_layers"        : dec_layers,
            "nheads"            : nheads,
            "camera_names"      : self.args.camera_names,
            "state_dim"         : self.args.state_dim, 
            "use_jupyter"       : self.args.use_jupyter,
        }

        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()

        self.optimizer = self.policy.configure_optimizers()

    def run(self):
        # save dataset stats
        if not os.path.isdir(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        stats_path = os.path.join(self.args.log_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"[TrainAct] Save dataset stats: {stats_path}")

        # train
        best_ckpt_info = self.train_bc()
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(self.args.log_dir, "policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"[TrainAct] Best ckpt, val loss {min_val_loss:.3f} @ epoch{best_epoch}")

    def train_bc(self):
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        for epoch in tqdm(range(self.args.num_epochs)):
            # validation
            with torch.inference_mode():
                self.policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(self.val_dataloader):
                    # print("val data " , data)
                    forward_dict = self.forward_pass(data)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (
                        epoch,
                        min_val_loss,
                        deepcopy(self.policy.state_dict()),
                    )
            summary_string = f"[TrainAct] val loss: {epoch_val_loss:.3f}"
            for k, v in epoch_summary.items():
                summary_string += f", {k}: {v.item():.3f}"
            print(summary_string)

            # training
            self.policy.train()
            self.optimizer.zero_grad()
            for batch_idx, data in enumerate(self.train_dataloader):
                # print("train data " , data)
                forward_dict = self.forward_pass(data)
                # backward
                loss = forward_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(
                train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
            )
            epoch_train_loss = epoch_summary["loss"]
            summary_string = f"[TrainAct] train loss: {epoch_train_loss:.3f}"
            for k, v in epoch_summary.items():
                summary_string += f", {k}: {v.item():.3f}"
            print(summary_string)

            if epoch % 100 == 0:
                ckpt_path = os.path.join(
                    self.args.log_dir,
                    f"policy_epoch_{epoch}_seed_{self.args.seed}.ckpt",
                )
                torch.save(self.policy.state_dict(), ckpt_path)
                self.plot_history(train_history, validation_history, epoch)

        ckpt_path = os.path.join(self.args.log_dir, "policy_last.ckpt")
        torch.save(self.policy.state_dict(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(
            self.args.log_dir, f"policy_epoch_{best_epoch}_seed_{self.args.seed}.ckpt"
        )
        torch.save(best_state_dict, ckpt_path)
        print(
            f"[TrainAct] Training finished: seed {self.args.seed}, val loss {min_val_loss:.3f} at epoch {best_epoch}"
        )

        # save training curves
        self.plot_history(train_history, validation_history, self.args.num_epochs)

        return best_ckpt_info

    def forward_pass(self, data):
        image_data, joint_data, action_data, is_pad = data
        # print("img data : ", image_data.shape, joint_data.shape, action_data.shape, is_pad.shape)
        image_data, joint_data, action_data, is_pad = (
            image_data.cuda(),
            joint_data.cuda(),
            action_data.cuda(),
            is_pad.cuda(),
        )
        return self.policy(
            joint_data, image_data, action_data, is_pad
        )  # TODO remove None

    def plot_history(self, train_history, validation_history, epoch):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(
                self.args.log_dir, f"train_val_{key}_seed_{self.args.seed}.png"
            )
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, epoch - 1, len(train_history)),
                train_values,
                label="train",
            )
            plt.plot(
                np.linspace(0, epoch - 1, len(validation_history)),
                val_values,
                label="validation",
            )
            # plt.ylim([-0.1, 1])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f"[TrainAct] Saved plots to {self.args.log_dir}")


if __name__ == "__main__":
    train = TrainAct()
    train.run()
