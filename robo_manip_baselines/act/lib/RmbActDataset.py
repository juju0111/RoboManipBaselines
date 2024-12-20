import numpy as np
import torch
from functools import lru_cache
from pathlib import Path
from torch.utils.data import DataLoader

import IPython

e = IPython.embed


@lru_cache(maxsize=128)
def load_array(dir_path, glob_pattern):
    globbed_list = list(dir_path.glob(glob_pattern))
    print("globbed_list : ", globbed_list)
    assert len(globbed_list) == 1
    return np.load(globbed_list[0])


class RmbActDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, is_sim, camera_names, norm_stats):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = is_sim
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(load_array(self.dataset_dir, "**/joints.npy"))

    def __getitem__(self, episode_id):
        sample_full_episode = False  # hardcode
        print("self.dataset_dir ", self.dataset_dir)
        original_action = load_array(self.dataset_dir, "**/actions.npy")[episode_id]
        original_action_shape = original_action.shape
        episode_len = original_action_shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        
        print("start_ts : ", start_ts)
        # get observation at start_ts only
        joint = load_array(self.dataset_dir, "**/joints.npy")[episode_id][start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            try:
                original_image = load_array(
                    self.dataset_dir, f"**/{cam_name}_images.npy"
                )
            except IndexError:
                print(f"self.dataset_dir:\t{self.dataset_dir}")
                print(f"cam_name:\t{cam_name}")
                raise
            image_dict[cam_name] = original_image[episode_id][start_ts]
        # get mask
        original_mask = load_array(self.dataset_dir, "**/masks.npy")[episode_id].astype(
            bool
        )
        # get all actions after and including start_ts
        if self.is_sim:
            action = original_action[start_ts:]
            action_len = episode_len - start_ts
            mask = original_mask[start_ts:]
        else:
            action = original_action[
                max(0, start_ts - 1) :
            ]  # hack, to make timesteps more aligned
            action_len = episode_len - max(
                0, start_ts - 1
            )  # hack, to make timesteps more aligned
            mask = original_mask[max(0, start_ts - 1) :]

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        padded_mask = np.zeros(episode_len, dtype=bool)
        padded_mask[:action_len] = mask
        is_pad = ~padded_mask

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        joint_data = torch.from_numpy(joint).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        joint_data = (joint_data - self.norm_stats["joint_mean"]) / self.norm_stats[
            "joint_std"
        ]

        return image_data, joint_data, action_data, is_pad


def get_norm_stats(train_dataset_dir, val_dataset_dir):
    all_joint_data = []
    all_action_data = []
    for dataset_dir in (train_dataset_dir, val_dataset_dir):
        try:
            joint = load_array(dataset_dir, "**/joints.npy")
            action = load_array(dataset_dir, "**/actions.npy")
        except IndexError:
            print(f"dataset_dir:\t{dataset_dir}")
            raise
        all_joint_data.append(torch.from_numpy(joint))
        all_action_data.append(torch.from_numpy(action))
    all_joint_data = torch.cat(all_joint_data)
    all_action_data = torch.cat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize joint data
    joint_mean = all_joint_data.mean(dim=[0, 1], keepdim=True)
    joint_std = all_joint_data.std(dim=[0, 1], keepdim=True)
    joint_std = torch.clip(joint_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "joint_mean": joint_mean.numpy().squeeze(),
        "joint_std": joint_std.numpy().squeeze(),
        "example_joint": joint,
    }

    return stats


def load_data(dataset_dir, is_sim, camera_names, batch_size_train, batch_size_val):
    print(f"\nData from: {dataset_dir}\n")
    dataset_dir = Path(dataset_dir)
    # obtain train test dataset dir
    train_dataset_dir = dataset_dir / "train"
    val_dataset_dir = dataset_dir / "test"

    # obtain normalization stats for joint and action
    norm_stats = get_norm_stats(train_dataset_dir, val_dataset_dir)

    # construct dataset and dataloader
    train_dataset = RmbActDataset(train_dataset_dir, is_sim, camera_names, norm_stats)
    val_dataset = RmbActDataset(val_dataset_dir, is_sim, camera_names, norm_stats)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
