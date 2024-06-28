# Action Chunking with Transformers (ACT)

## Install

Install [SARNN](../sarnn) according to [here](../sarnn/README.md).

Install [ACT](https://github.com/tonyzhaozh/act).
``` console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ cd third_party/act
$ pip install torchvision torch pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 \
 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython
$ cd detr
$ pip install -e .
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── env0
│   ├── UR5eCableEnv_env0_000.npz
│   └── UR5eCableEnv_env0_006.npz
├── env1
│   ├── UR5eCableEnv_env1_001.npz
│   └── UR5eCableEnv_env1_007.npz
├── env2
│   ├── UR5eCableEnv_env2_002.npz
│   └── UR5eCableEnv_env2_008.npz
├── env3
│   ├── UR5eCableEnv_env3_003.npz
│   └── UR5eCableEnv_env3_009.npz
├── env4
│   ├── UR5eCableEnv_env4_004.npz
│   └── UR5eCableEnv_env4_010.npz
└── env5
    ├── UR5eCableEnv_env5_005.npz
    └── UR5eCableEnv_env5_011.npz
```

Make numpy files in each of `train` (for training) and `test` directories (for validation).

```console
$ python ../utils/make_dataset.py --in_dir ./data/teleop_data_sample --out_dir ./data/learning_data_sample --train_keywords env0 env1 env4 env5 --test_keywords env2 env3 --nproc `nproc` --skip 1
```

Visualize the generated data (optional).

```console
$ python ../utils/check_data.py --in_dir ./data/learning_data_sample --idx 0
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ python ./bin/train.py \
--dataset_dir ./data/learning_data_sample --ckpt_dir ./log/YEAR_DAY_TIME --task_name sim_ur5ecable \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 1000 --lr 1e-5 \
--seed 0
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ python ./bin/rollout.py \
--ckpt_dir ./log/YEAR_DAY_TIME --ckpt_name policy_best.ckpt --task_name sim_ur5ecable \
--policy_class ACT --chunk_size 100 --num_epochs 0 \
--seed 0 \
--pole-pos-idx 0
```