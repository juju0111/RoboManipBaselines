from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutIsaacUR5eChain


class RolloutActIsaacUR5eChain(RolloutAct, RolloutIsaacUR5eChain):
    pass


if __name__ == "__main__":
    rollout = RolloutActIsaacUR5eChain()
    rollout.run()
