import os
import numpy as np
import cv2
from enum import Enum
import gymnasium as gym
import pinocchio as pin
import mujoco

# https://github.com/opencv/opencv/issues/21326
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionManager(object):
    """
    Motion manager for robot arm and gripper.

    The manager computes forward and inverse kinematics using Pinocchio, a robot modeling library.
    """
    def __init__(self, env, root_body_name="ur5e_root_frame"):
        self.env = env

        # Setup pinocchio model and data
        root_body = env.unwrapped.model.body(root_body_name)
        root_se3 = pin.SE3(pin.Quaternion(root_body.quat[[1, 2, 3, 0]]), root_body.pos)
        self.pin_model = pin.buildModelFromUrdf(self.env.unwrapped.urdf_path)
        self.pin_model.jointPlacements[1] = root_se3.act(self.pin_model.jointPlacements[1]) # Set root link pose
        self.pin_data = self.pin_model.createData()
        self.pin_data_obs = self.pin_model.createData()

        # Setup arm
        self.joint_pos = self.env.unwrapped.init_qpos[:6].copy()
        self.eef_joint_id = 6
        pin.forwardKinematics(self.pin_model, self.pin_data, self.joint_pos)
        self._original_target_se3 = self.pin_data.oMi[self.eef_joint_id].copy()
        self.target_se3 = self._original_target_se3.copy()

        # Setup gripper
        self._gripper_pos = 0.0
        self.gripper_action_idx = 6

    def reset(self):
        """Reset states of arm and gripper."""
        self.joint_pos = self.env.unwrapped.init_qpos[:6].copy()
        self.target_se3 = self._original_target_se3.copy()
        self.gripper_pos = 0.0

    def inverseKinematics(self):
        """Solve inverse kinematics."""
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        error_se3 = self.current_se3.actInv(self.target_se3)
        error_vec = pin.log(error_se3).vector # in joint frame
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, self.joint_pos, self.eef_joint_id) # in joint frame
        J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
        damping_scale = 1e-6
        delta_joint_pos = -1 * J.T.dot(np.linalg.solve(
            J.dot(J.T) + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6), error_vec))
        self.joint_pos = pin.integrate(self.pin_model, self.joint_pos, delta_joint_pos)
        pin.forwardKinematics(self.pin_model, self.pin_data, self.joint_pos)

    def setRelativeTargetSE3(self, delta_pos=None, delta_rpy=None):
        """Set the target pose of the end-effector relatively."""
        if delta_pos is not None:
            self.target_se3.translation += delta_pos
        if delta_rpy is not None:
            self.target_se3.rotation = np.matmul(pin.rpy.rpyToMatrix(*delta_rpy), self.target_se3.rotation)

    def drawMarkers(self):
        """Draw markers of the current and target poses of the end-effector to viewer."""
        self.env.unwrapped.mujoco_renderer.viewer.add_marker(
            pos=self.target_se3.translation,
            mat=self.target_se3.rotation,
            label="",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.02, 0.02, 0.03),
            rgba=(0, 1, 0, 0.5))
        self.env.unwrapped.mujoco_renderer.viewer.add_marker(
            pos=self.current_se3.translation,
            mat=self.current_se3.rotation,
            label="",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.02, 0.02, 0.03),
            rgba=(1, 0, 0, 0.5))

    def getAction(self):
        """Get action for Gym."""
        return np.concatenate([self.joint_pos, [self.gripper_pos]])

    def getJointPos(self, obs):
        """Get joint position from observation."""
        arm_qpos = obs[0:6]
        gripper_pos = np.rad2deg(obs[12:16].mean(keepdims=True)) / 45.0 * 255.0
        return np.concatenate((arm_qpos, gripper_pos))

    def getJointVel(self, obs):
        """Get joint velocity from observation."""
        arm_qvel = obs[6:12]
        # Set zero as a dummy because the joint velocity of gripper cannot be obtained
        gripper_vel = np.zeros(1)
        return np.concatenate((arm_qvel, gripper_vel))

    def getMeasuredEef(self, obs):
        """Get measured end-effector pose (tx, ty, tz, rx, ry, rz, rw) from observation."""
        arm_qpos = obs[0:6]
        pin.forwardKinematics(self.pin_model, self.pin_data_obs, arm_qpos)
        measured_se3 = self.pin_data_obs.oMi[self.eef_joint_id]
        return np.concatenate([measured_se3.translation, pin.Quaternion(measured_se3.rotation).coeffs()])

    def getCommandEef(self):
        """Get command end-effector pose (tx, ty, tz, rx, ry, rz, rw)."""
        return np.concatenate([self.target_se3.translation, pin.Quaternion(self.target_se3.rotation).coeffs()])

    @property
    def current_se3(self):
        """Get the current pose of the end-effector."""
        return self.pin_data.oMi[self.eef_joint_id]

    @property
    def gripper_pos(self):
        """Get the target gripper position."""
        return self._gripper_pos

    @gripper_pos.setter
    def gripper_pos(self, new_gripper_pos):
        """Set the target gripper position."""
        self._gripper_pos = np.clip(new_gripper_pos,
                                    self.env.action_space.low[self.gripper_action_idx],
                                    self.env.action_space.high[self.gripper_action_idx])

class RecordStatus(Enum):
    """Status for recording."""
    INITIAL = 0
    PRE_REACH = 1
    REACH = 2
    GRASP = 3
    TELEOP = 4
    END = 5

class RecordKey(Enum):
    """Data key for recording."""
    TIME = 0
    JOINT_POS = 1
    JOINT_VEL = 2
    FRONT_RGB_IMAGE = 3
    SIDE_RGB_IMAGE = 4
    HAND_RGB_IMAGE = 5
    FRONT_DEPTH_IMAGE = 6
    SIDE_DEPTH_IMAGE = 7
    HAND_DEPTH_IMAGE = 8
    WRENCH = 9
    MEASURED_EEF = 10
    COMMAND_EEF = 11
    ACTION = 12

    def key(self):
        """Get the key of the dictionary."""
        return self.name.lower()

class RecordManager(object):
    """Recording manager for demonstrations by teleoperation."""

    def __init__(self, env):
        self.env = env

        self.data_idx = 0
        self.world_idx = 0
        self.world_info = {}

        self.camera_info = {}

        self.reset()

    def reset(self):
        """Reset recording."""
        self.status = RecordStatus(0)

        self.data_seq = {}
        for record_key in RecordKey:
            self.data_seq[record_key.key()] = []

    def appendSingleData(self, record_key, data):
        """Append a single data to the data sequence."""
        self.data_seq[record_key.key()].append(data)

    def getSingleData(self, record_key, time_idx):
        """Get single data from the data sequence."""
        key = record_key.key()
        data = self.data_seq[key][time_idx]
        if "rgb" in key:
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_COLOR)
        elif "depth" in key:
            if data.ndim == 1:
                data = cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED)
        return data

    def getData(self, record_key):
        """Get data."""
        key = record_key.key()
        data_seq = self.data_seq[key]
        if "rgb" in key:
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_COLOR) for data in data_seq])
        elif "depth" in key:
            if data_seq[0].ndim == 1:
                data_seq = np.array([cv2.imdecode(data, flags=cv2.IMREAD_UNCHANGED) for data in data_seq])
        return data_seq

    def compressData(self, record_key, compress_flag):
        """Compress data."""
        key = record_key.key()
        for time_idx, data in enumerate(self.data_seq[key]):
            if compress_flag == "jpg":
                self.data_seq[key][time_idx] = cv2.imencode(".jpg", data, (cv2.IMWRITE_JPEG_QUALITY, 95))[1]
            elif compress_flag == "exr":
                self.data_seq[key][time_idx] = cv2.imencode(".exr", data)[1]

    def saveData(self, filename):
        """Save data."""
        # If each element has a different shape, save it as an object array
        for key in self.data_seq.keys():
            if isinstance(self.data_seq[key], list) and \
               len({data.shape if isinstance(data, np.ndarray) else None for data in self.data_seq[key]}) > 1:
                self.data_seq[key] = np.array(self.data_seq[key], dtype=object)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **self.data_seq, **self.world_info, **self.camera_info)
        self.data_idx += 1

    def loadData(self, filename):
        """Load data."""
        npz_data = np.load(filename, allow_pickle=True)
        self.data_seq = dict()
        for key in npz_data.keys():
            self.data_seq[key] = np.copy(npz_data[key])

    def goToNextStatus(self):
        """Go to the next status."""
        self.status = RecordStatus((self.status.value + 1) % len(RecordStatus))

    def getStatusImage(self):
        """Get the image corresponding to the current status."""
        status_image = np.zeros((50, 160, 3), dtype=np.uint8)
        if self.status == RecordStatus.INITIAL:
            status_image[:, :] = np.array([200, 255, 200])
        elif self.status in {RecordStatus.PRE_REACH, RecordStatus.REACH, RecordStatus.GRASP}:
            status_image[:, :] = np.array([255, 255, 200])
        elif self.status == RecordStatus.TELEOP:
            status_image[:, :] = np.array([255, 200, 200])
        elif self.status == RecordStatus.END:
            status_image[:, :] = np.array([200, 200, 255])
        else:
            raise ValueError("Unknown status: {}".format(self.status))
        cv2.putText(status_image, self.status.name, (5, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        return status_image

    def setupSimWorld(self, world_idx=None):
        """Setup the simulation world."""
        if world_idx is None:
            kwargs = {"cumulative_idx": self.data_idx}
        else:
            kwargs = {"world_idx": world_idx}
        self.world_idx = self.env.unwrapped.modify_world(**kwargs)
        self.world_info = {"world_idx": self.world_idx}

    def setupCameraInfo(self, camera_names, depth_keys):
        """Set camera info."""
        for camera_name, depth_key in zip(camera_names, depth_keys):
            self.camera_info[depth_key.key() + "_fovy"] = self.env.unwrapped.model.cam(camera_name).fovy[0]

    @property
    def status_elapsed_duration(self):
        """Get the elapsed duration of the current status."""
        return self.env.unwrapped.data.time - self.status_start_time

    @property
    def status(self):
        """Get the status."""
        return self._status

    @status.setter
    def status(self, new_status):
        """Set the status."""
        self._status = new_status
        if self.env is None:
            self.status_start_time = 0.0
        else:
            self.status_start_time = self.env.unwrapped.data.time

def convertDepthImageToColorImage(image):
    """Convert depth image (float type) to color image (uint8 type)."""
    image = (255 * ((image - image.min()) / (image.max() - image.min()))).astype(np.uint8)
    return cv2.merge((image,) * 3)

def convertDepthImageToPointCloud(depth_image, fovy, rgb_image=None, dist_thre=None):
    """Convert depth image (float type) to point cloud (array of 3D position)."""
    focal_scaling = (1.0 / np.tan(np.deg2rad(fovy) / 2.0)) * depth_image.shape[0] / 2.0
    xyz_array = np.array([(i, j) for i in range(depth_image.shape[0]) for j in range(depth_image.shape[1])], dtype=np.float32)
    xyz_array = (xyz_array - 0.5 * np.array(depth_image.shape[:2], dtype=np.float32)) / focal_scaling
    xyz_array *= depth_image.flatten()[:, np.newaxis]
    xyz_array = np.hstack((xyz_array[:, [1, 0]], depth_image.flatten()[:, np.newaxis]))
    if dist_thre:
        dist_thre_indices = np.argwhere(depth_image.flatten() < dist_thre)[:, 0]
        xyz_array = xyz_array[dist_thre_indices]
        if rgb_image is not None:
            rgb_array = rgb_image.reshape(-1, 3)[dist_thre_indices]
    if rgb_image is None:
        return xyz_array
    else:
        rgb_array = rgb_array.astype(np.float32) / 255.0
        return xyz_array, rgb_array