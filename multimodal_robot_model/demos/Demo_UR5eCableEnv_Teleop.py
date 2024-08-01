import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
import pyspacemouse
from Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[
      {"name": "front", "size": (480, 640)},
      {"name": "side", "size": (480, 640)},
      {"name": "hand", "size": (480, 640)},
  ]
)
obs, info = env.reset(seed=42)

# Setup motion manager
motion_manager = MotionManager(env)

# Setup record manager
record_manager = RecordManager(env)

# Setup spacemouse
pyspacemouse.open()

reset = True
while True:
    # Reset
    if reset:
        motion_manager.reset()
        record_manager.reset()
        record_manager.setupSimWorld()
        obs, info = env.reset()
        print("== [UR5eCableEnv] data_idx: {}, world_idx: {} ==".format(record_manager.data_idx, record_manager.world_idx))
        print("- Press space key to start automatic grasping.")
        reset = False

    # Read spacemouse
    spacemouse_state = pyspacemouse.read()

    # Set arm command
    if record_manager.status == RecordStatus.PRE_REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 1.02 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 0.995 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.TELEOP:
        pos_scale = 1e-2
        delta_pos = pos_scale * np.array([-1.0 * spacemouse_state.y, spacemouse_state.x, spacemouse_state.z])
        rpy_scale = 5e-3
        delta_rpy = rpy_scale * np.array([-1.0 * spacemouse_state.roll, -1.0 * spacemouse_state.pitch, -2.0 * spacemouse_state.yaw])
        motion_manager.setRelativeTargetSE3(delta_pos, delta_rpy)

    # Set gripper command
    if record_manager.status == RecordStatus.GRASP:
        motion_manager.gripper_pos = env.action_space.high[6]
    elif record_manager.status == RecordStatus.TELEOP:
        gripper_scale = 5.0
        if spacemouse_state.buttons[0] > 0 and spacemouse_state.buttons[1] <= 0:
            motion_manager.gripper_pos += gripper_scale
        elif spacemouse_state.buttons[1] > 0 and spacemouse_state.buttons[0] <= 0:
            motion_manager.gripper_pos -= gripper_scale

    # Draw markers
    motion_manager.drawMarkers()

    # Solve IK
    motion_manager.inverseKinematics()

    # Get action
    action = motion_manager.getAction()

    # Record data
    if record_manager.status == RecordStatus.TELEOP:
        record_manager.appendSingleData(RecordKey.TIME, record_manager.status_elapsed_duration)
        record_manager.appendSingleData(RecordKey.JOINT_POS, motion_manager.getJointPos(obs))
        record_manager.appendSingleData(RecordKey.JOINT_VEL, motion_manager.getJointVel(obs))
        record_manager.appendSingleData(RecordKey.FRONT_RGB_IMAGE, info["rgb_images"]["front"])
        record_manager.appendSingleData(RecordKey.SIDE_RGB_IMAGE, info["rgb_images"]["side"])
        record_manager.appendSingleData(RecordKey.HAND_RGB_IMAGE, info["rgb_images"]["hand"])
        record_manager.appendSingleData(RecordKey.FRONT_DEPTH_IMAGE, info["depth_images"]["front"])
        record_manager.appendSingleData(RecordKey.SIDE_DEPTH_IMAGE, info["depth_images"]["side"])
        record_manager.appendSingleData(RecordKey.HAND_DEPTH_IMAGE, info["depth_images"]["hand"])
        record_manager.appendSingleData(RecordKey.WRENCH, obs[16:])
        record_manager.appendSingleData(RecordKey.ACTION, action)

    # Step environment
    obs, _, _, _, info = env.step(action)

    # Draw images
    window_images = []
    for camera_name in ("front", "side", "hand"):
        image_size = env.unwrapped.cameras[camera_name]["size"]
        image_ratio = image_size[1] / image_size[0]
        window_images.append(cv2.resize(info["rgb_images"][camera_name], (224, int(224 / image_ratio))))
    window_images.append(record_manager.getStatusImage())
    window_image = np.concatenate(window_images)
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)

    # Manage status
    if record_manager.status == RecordStatus.INITIAL:
        if key == 32: # space key
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.PRE_REACH:
        pre_reach_duration = 0.7 # [s]
        if record_manager.status_elapsed_duration > pre_reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.REACH:
        reach_duration = 0.3 # [s]
        if record_manager.status_elapsed_duration > reach_duration:
            record_manager.goToNextStatus()
            print("- Press space key to start teleoperation after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        if key == 32: # space key
            record_manager.goToNextStatus()
            print("- Press space key to finish teleoperation.")
    elif record_manager.status == RecordStatus.TELEOP:
        if key == 32: # space key
            print("- Press the 's' key if the teleoperation succeeded, or the 'f' key if it failed. (duration: {:.1f} [s])".format(
                record_manager.status_elapsed_duration))
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.END:
        if key == ord("s"):
            # Save data
            filename = "teleop_data/env{:0>1}/UR5eCableEnv_env{:0>1}_{:0>3}.npz".format(
                record_manager.world_idx, record_manager.world_idx, record_manager.data_idx)
            record_manager.saveData(filename)
            print("- Teleoperation succeeded: Save the data as {}".format(filename))
            reset = True
        elif key == ord("f"):
            print("- Teleoperation failed: Reset without saving")
            reset = True
    if key == 27: # escape key
        break

# env.close()
