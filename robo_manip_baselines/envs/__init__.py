from gymnasium.envs.registration import register

# Mujoco
## UR5e
register(
    id="robo_manip_baselines/MujocoUR5eCableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eCableEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eRingEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eRingEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eParticleEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eParticleEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eClothEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eClothEnv",
)

## Xarm7
register(
    id="robo_manip_baselines/MujocoXarm7CableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoXarm7CableEnv",
)

register(
    id="robo_manip_baselines/MujocoXarm7RingEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoXarm7RingEnv",
)

## Aloha
register(
    id="robo_manip_baselines/MujocoAlohaCableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoAlohaCableEnv",
)

# Isaac
register(
    id="robo_manip_baselines/IsaacUR5eChainEnv-v0",
    entry_point="robo_manip_baselines.envs.isaac:IsaacUR5eChainEnv",
)
register(
    id="robo_manip_baselines/IsaacUR5eCabinetEnv-v0",
    entry_point="robo_manip_baselines.envs.isaac:IsaacUR5eCabinetEnv",
)

# Real
## UR5e
register(
    id="robo_manip_baselines/RealUR5eDemoEnv-v0",
    entry_point="robo_manip_baselines.envs.real.ur5e:RealUR5eDemoEnv",
)

## Xarm7
register(
    id="robo_manip_baselines/RealXarm7DemoEnv-v0",
    entry_point="robo_manip_baselines.envs.real.xarm7:RealXarm7DemoEnv",
)
