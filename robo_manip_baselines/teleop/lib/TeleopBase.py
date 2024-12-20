from abc import ABCMeta, abstractmethod
import argparse
import time
import datetime
import numpy as np
import cv2
import matplotlib.pylab as plt
import pyspacemouse
from robo_manip_baselines.common import (
    MotionManager,
    MotionStatus,
    DataKey,
    DataManager,
    convertDepthImageToColorImage,
    convertDepthImageToPointCloud,
)


class TeleopBase(metaclass=ABCMeta):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--demo_name", type=str, default=None, help="demonstration name"
        )
        parser.add_argument(
            "--enable_3d_plot", action="store_true", help="whether to enable 3d plot"
        )
        parser.add_argument(
            "--compress_rgb", type=int, default=1, help="whether to compress rgb image"
        )
        parser.add_argument(
            "--compress_depth",
            type=int,
            default=0,
            help="whether to compress depth image (slow)",
        )
        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            help="list of world indexes (if not given, loop through all world indicies)",
        )
        parser.add_argument(
            "--replay_log",
            type=str,
            default=None,
            help="log file path when replay log motion",
        )
        self.args = parser.parse_args()

        # Setup gym environment
        self.setup_env()
        self.env.reset(seed=42)

        # Setup motion manager
        MotionManagerClass = getattr(self, "MotionManagerClass", MotionManager)
        self.motion_manager = MotionManagerClass(self.env)

        # Setup data manager
        DataManagerClass = getattr(self, "DataManagerClass", DataManager)
        self.data_manager = DataManagerClass(self.env, demo_name=self.demo_name)
        self.data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()

        # Setup 3D plot
        if self.args.enable_3d_plot:
            plt.rcParams["keymap.quit"] = ["q", "escape"]
            self.fig, self.ax = plt.subplots(
                len(self.env.unwrapped.camera_names),
                1,
                subplot_kw=dict(projection="3d"),
            )
            self.fig.tight_layout()
            self.point_cloud_scatter_list = [None] * len(
                self.env.unwrapped.camera_names
            )

        # Command configuration
        self._spacemouse_connected = False
        self.command_pos_scale = 1e-2
        self.command_rpy_scale = 5e-3
        self.gripper_scale = 5.0

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            # Reset
            if self.reset_flag:
                self.motion_manager.reset()
                if self.args.replay_log is None:
                    self.data_manager.reset()
                    if self.args.world_idx_list is None:
                        world_idx = None
                    else:
                        world_idx = self.args.world_idx_list[
                            self.data_manager.data_idx % len(self.args.world_idx_list)
                        ]
                else:
                    self.data_manager.load_data(self.args.replay_log)
                    print("- Load teleoperation data: {}".format(self.args.replay_log))
                    world_idx = self.data_manager.get_data("world_idx").tolist()
                self.data_manager.setup_sim_world(world_idx)
                obs, info = self.env.reset()
                print(
                    "[{}] data_idx: {}, world_idx: {}".format(
                        self.demo_name,
                        self.data_manager.data_idx,
                        self.data_manager.world_idx,
                    )
                )
                print("- Press the 'n' key to start automatic grasping.")
                self.reset_flag = False

            # Read spacemouse
            if self.data_manager.status == MotionStatus.TELEOP:
                # Empirically, you can call read repeatedly to get the latest device status
                for i in range(10):
                    self.spacemouse_state = pyspacemouse.read()

            # Get action
            if self.args.replay_log is not None and self.data_manager.status in (
                MotionStatus.TELEOP,
                MotionStatus.END,
            ):
                action = self.data_manager.get_single_data(
                    DataKey.COMMAND_JOINT_POS, self.teleop_time_idx
                )
            else:
                # Set commands
                self.set_arm_command()
                self.set_gripper_command()

                # Solve IK
                self.motion_manager.draw_markers()
                self.motion_manager.inverse_kinematics()

                action = self.motion_manager.get_action()

            # Record data
            if (
                self.data_manager.status == MotionStatus.TELEOP
                and self.args.replay_log is None
            ):
                self.data_manager.append_single_data(
                    DataKey.TIME, self.data_manager.status_elapsed_duration
                )
                self.data_manager.append_single_data(
                    DataKey.MEASURED_JOINT_POS, self.motion_manager.get_joint_pos(obs)
                )
                self.data_manager.append_single_data(DataKey.COMMAND_JOINT_POS, action)
                self.data_manager.append_single_data(
                    DataKey.MEASURED_JOINT_VEL, self.motion_manager.get_joint_vel(obs)
                )
                self.data_manager.append_single_data(
                    DataKey.MEASURED_EEF_POSE, self.motion_manager.get_measured_eef(obs)
                )
                self.data_manager.append_single_data(
                    DataKey.COMMAND_EEF_POSE, self.motion_manager.get_command_eef()
                )
                self.data_manager.append_single_data(
                    DataKey.MEASURED_EEF_WRENCH, self.motion_manager.get_eef_wrench(obs)
                )
                for camera_name in self.env.unwrapped.camera_names:
                    self.data_manager.append_single_data(
                        DataKey.get_rgb_image_key(camera_name),
                        info["rgb_images"][camera_name],
                    )
                    self.data_manager.append_single_data(
                        DataKey.get_depth_image_key(camera_name),
                        info["depth_images"][camera_name],
                    )

            # Step environment
            obs, _, _, _, info = self.env.step(action)

            # Draw images
            self.drawImage(info)

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.drawPointCloud(info)

            # Manage status
            self.manage_status()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.data_manager.status == MotionStatus.TELEOP:
                iteration_duration_list.append(iteration_duration)
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        print("- Statistics on teleoperation")
        if len(iteration_duration_list) > 0:
            iteration_duration_list = np.array(iteration_duration_list)
            print(
                f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_list.mean():.2f}"
            )
            print(
                "  - Iteration duration [s] | "
                f"mean: {iteration_duration_list.mean():.3f}, std: {iteration_duration_list.std():.3f} "
                f"min: {iteration_duration_list.min():.3f}, max: {iteration_duration_list.max():.3f}"
            )

        # self.env.close()

    @abstractmethod
    def setup_env(self):
        pass

    def set_arm_command(self):
        if self.data_manager.status == MotionStatus.TELEOP:
            delta_pos = self.command_pos_scale * np.array(
                [
                    -1.0 * self.spacemouse_state.y,
                    self.spacemouse_state.x,
                    self.spacemouse_state.z,
                ]
            )
            delta_rpy = self.command_rpy_scale * np.array(
                [
                    -1.0 * self.spacemouse_state.roll,
                    -1.0 * self.spacemouse_state.pitch,
                    -2.0 * self.spacemouse_state.yaw,
                ]
            )
            self.motion_manager.set_relative_target_se3(delta_pos, delta_rpy)

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[
                self.env.unwrapped.gripper_action_idx
            ]
        elif self.data_manager.status == MotionStatus.TELEOP:
            if (
                self.spacemouse_state.buttons[0] > 0
                and self.spacemouse_state.buttons[-1] <= 0
            ):
                self.motion_manager.gripper_pos += self.gripper_scale
            elif (
                self.spacemouse_state.buttons[-1] > 0
                and self.spacemouse_state.buttons[0] <= 0
            ):
                self.motion_manager.gripper_pos -= self.gripper_scale

    def drawImage(self, info):
        status_image = self.data_manager.get_status_image()
        rgb_images = []
        depth_images = []
        for camera_name in self.env.unwrapped.camera_names:
            rgb_image = info["rgb_images"][camera_name]
            image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            resized_image_width = status_image.shape[1] / 2
            resized_image_size = (
                int(resized_image_width),
                int(resized_image_width / image_ratio),
            )
            rgb_images.append(cv2.resize(rgb_image, resized_image_size))
            depth_image = convertDepthImageToColorImage(
                info["depth_images"][camera_name]
            )
            depth_images.append(cv2.resize(depth_image, resized_image_size))
        window_image = cv2.vconcat(
            (
                cv2.hconcat((cv2.vconcat(rgb_images), cv2.vconcat(depth_images))),
                status_image,
            )
        )
        cv2.namedWindow(
            "image",
            flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL),
        )
        cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    def drawPointCloud(self, info):
        dist_thre_list = (3.0, 3.0, 0.8)  # [m]
        for camera_idx, camera_name in enumerate(self.env.unwrapped.camera_names):
            point_cloud_skip = 10
            small_depth_image = info["depth_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            small_rgb_image = info["rgb_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            fovy = self.data_manager.camera_info[
                DataKey.get_depth_image_key(camera_name) + "_fovy"
            ]
            xyz_array, rgb_array = convertDepthImageToPointCloud(
                small_depth_image,
                fovy=fovy,
                rgb_image=small_rgb_image,
                dist_thre=dist_thre_list[camera_idx],
            )
            if self.point_cloud_scatter_list[camera_idx] is None:

                def get_min_max(v_min, v_max):
                    return (
                        0.75 * v_min + 0.25 * v_max,
                        0.25 * v_min + 0.75 * v_max,
                    )

                self.ax[camera_idx].view_init(elev=-90, azim=-90)
                self.ax[camera_idx].set_xlim(
                    *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
                )
                self.ax[camera_idx].set_ylim(
                    *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
                )
                self.ax[camera_idx].set_zlim(
                    *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
                )
            else:
                self.point_cloud_scatter_list[camera_idx].remove()
            self.ax[camera_idx].axis("off")
            self.ax[camera_idx].set_box_aspect(xyz_array.ptp(axis=0))
            self.point_cloud_scatter_list[camera_idx] = self.ax[camera_idx].scatter(
                xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
            )
        plt.draw()
        plt.pause(0.001)

    def manage_status(self):
        key = cv2.waitKey(1)
        if self.data_manager.status == MotionStatus.INITIAL:
            if key == ord("n"):
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.PRE_REACH:
            pre_reach_duration = 0.7  # [s]
            if self.data_manager.status_elapsed_duration > pre_reach_duration:
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.REACH:
            reach_duration = 0.3  # [s]
            if self.data_manager.status_elapsed_duration > reach_duration:
                print(
                    "- Press the 'n' key to start teleoperation after the gripper is closed."
                )
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.GRASP:
            if key == ord("n"):
                # Setup spacemouse
                if (self.args.replay_log is None) and (not self._spacemouse_connected):
                    self._spacemouse_connected = True
                    pyspacemouse.open()
                self.teleop_time_idx = 0
                if self.args.replay_log is None:
                    print("- Press the 'n' key to finish teleoperation.")
                else:
                    print("- Start to replay the log motion.")
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.TELEOP:
            self.teleop_time_idx += 1
            if self.args.replay_log is None:
                if key == ord("n"):
                    print(
                        "- Press the 's' key if the teleoperation succeeded,"
                        " or the 'f' key if it failed. (duration: {:.1f} [s])".format(
                            self.data_manager.status_elapsed_duration
                        )
                    )
                    self.data_manager.go_to_next_status()
            else:
                if self.teleop_time_idx == len(self.data_manager.get_data("time")):
                    self.teleop_time_idx -= 1
                    print(
                        "- The log motion has finished replaying. Press the 'n' key to exit."
                    )
                    self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.END:
            if self.args.replay_log is None:
                if key == ord("s"):
                    # Save data
                    self.save_data()
                    self.reset_flag = True
                elif key == ord("f"):
                    print("- Teleoperation failed: Reset without saving")
                    self.reset_flag = True
            else:
                if key == ord("n"):
                    self.quit_flag = True
        if key == 27:  # escape key
            self.quit_flag = True

    def save_data(self):
        filename = (
            "teleop_data/{}_{:%Y%m%d_%H%M%S}/env{:0>1}/{}_env{:0>1}_{:0>3}.npz".format(
                self.demo_name,
                self.datetime_now,
                self.data_manager.world_idx,
                self.demo_name,
                self.data_manager.world_idx,
                self.data_manager.data_idx,
            )
        )
        if self.args.compress_rgb:
            print("- Compress rgb images")
            for camera_name in self.env.unwrapped.camera_names:
                self.data_manager.compress_data(
                    DataKey.get_rgb_image_key(camera_name), "jpg"
                )
        if self.args.compress_depth:
            print("- Compress depth images")
            for camera_name in self.env.unwrapped.camera_names:
                self.data_manager.compress_data(
                    DataKey.get_depth_image_key(camera_name), "exr"
                )
        self.data_manager.save_data(filename)
        print("- Teleoperation succeeded: Save the data as {}".format(filename))
