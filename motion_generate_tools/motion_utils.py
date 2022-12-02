import json
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from data_loaders.humanml.common import quaternion

from mdm.data_loaders import tensors
from mdm.data_loaders.humanml.common import skeleton
from mdm.data_loaders.humanml.scripts import motion_process
from mdm.utils import dist_util

WORK_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(WORK_PATH, '..', 'data'))


def config_logging():
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d | %(levelname).4s | %(module)s | %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def disable_tqdm_globally():
    from functools import partialmethod

    from tqdm import tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class MotionOffsets:
    initial_position: np.ndarray
    initial_quaternion: np.ndarray


class HumanML3DConverter:
    @staticmethod
    def initialize_motion_process(t2m_tgt_skel: np.ndarray, t2m_raw_offsets: np.ndarray, t2m_kinematic_chain: List[List[int]]):
        # initialize global variables
        example_data = t2m_tgt_skel
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)

        # Lower legs
        motion_process.l_idx1, motion_process.l_idx2 = 5, 8
        # Right/Left foot
        motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        motion_process.face_joint_indx = [2, 1, 17, 16]
        # l_hip, r_hip
        motion_process.r_hip, motion_process.l_hip = 2, 1
        motion_process.joints_num = 22

        motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        motion_process.kinematic_chain = t2m_kinematic_chain
        motion_process.tgt_skel = skeleton.Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, 'cpu')

        motion_process.tgt_offsets = motion_process.tgt_skel.get_offsets_joints(example_data[0])

    def __init__(self, max_frames: int, features_mean: np.ndarray, features_std: np.ndarray):
        self.max_frames = max_frames
        self.mean: np.ndarray = features_mean
        self.std: np.ndarray = features_std

    def uniform_skeleton(self, motion: np.ndarray) -> np.ndarray:
        """
        Uniform skeleton.

        Parameters
        ----------
        motion : NDArray[Shape[",22,3"], Float]

        Returns
        -------
        uniformed_motion : NDArray[Shape[",22,3"], Float]
        """
        return motion_process.uniform_skeleton(motion, motion_process.tgt_offsets)

    def get_offsets(self, motion: np.ndarray) -> MotionOffsets:
        """
        Get offsets.

        Parameters
        ----------
        motion : NDArray[Shape[",22,3"], Float]

        Returns
        -------
        motion_offsets : MotionOffsets
        """

        # XYZ at origin
        root_pos_init = motion[0]
        root_pose_init_xyz = root_pos_init[0] * np.array([1, 0, 1])
        root_pose_init_xyz[1] = motion.min(axis=0).min(axis=0)[1]  # Put on Floor

        # All initially face Z+
        r_hip, l_hip, sdr_r, sdr_l = motion_process.face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = quaternion.qbetween_np(forward_init, target)
        return MotionOffsets(root_pose_init_xyz, root_quat_init)

    def apply_offsets(self, motion: np.ndarray, offsets: MotionOffsets) -> np.ndarray:
        """
        Apply offsets.

        Parameters
        ----------
        motion: NDArray[Shape[",22,3"], Float]
        offsets: MotionOffsets

        Returns
        -------
        offseted_motion : NDArray[Shape[",22,3"], Float]
        """
        motion = motion - offsets.initial_position
        motion = quaternion.qrot_np(np.ones(motion.shape[:-1] + (4,)) * offsets.initial_quaternion, motion)
        return motion

    def unapply_offsets(self, motion: np.ndarray, offsets: MotionOffsets) -> np.ndarray:
        """
        Unapply offsets.

        Parameters
        ----------
        motion: NDArray[Shape[",22,3"], Float]
        offsets: MotionOffsets

        Returns
        -------
        unoffseted_motion : NDArray[Shape[",22,3"], Float]
        """
        motion = quaternion.qrot_np(np.ones(motion.shape[:-1] + (4,)) * quaternion.qinv_np(offsets.initial_quaternion), motion)
        motion = motion + offsets.initial_position
        return motion

    def to_features(self, motion: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        To features.

        Parameters
        ----------
        motion: NDArray[Shape[",22,3"], Float]

        Returns
        -------
        feature : Tensor[Shape["1,263,1,"]]
        """

        features = motion_process.extract_features(
            motion.copy(),
            0.002,
            motion_process.n_raw_offsets,
            motion_process.kinematic_chain,
            motion_process.face_joint_indx,
            motion_process.fid_r, motion_process.fid_l
        )
        motion_length = len(features)
        features = (features - self.mean) / self.std
        features = np.concatenate([
            features,
            np.zeros((self.max_frames - motion_length, features.shape[1]))
        ], axis=0)
        features = tensors.collate_tensors([
            torch.tensor(features.T).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]
        ])
        features = features.to(dist_util.dev())
        return features, motion_length

    def to_motion(self, features: torch.Tensor, motion_length: int) -> np.ndarray:
        """
        To motion.

        Parameters
        ----------
        feature : Tensor[Shape["1,263,1,"]]

        Returns
        -------
        motion: NDArray[Shape[",22,3"], Float]
        """

        features = ((features.cpu().permute(0, 2, 3, 1)) * self.std + self.mean).float()
        features = motion_process.recover_from_ric(features, motion_process.joints_num)
        features = features.view(-1, *features.shape[2:]).permute(0, 2, 3, 1)
        motion: np.ndarray = features.cpu().numpy()[0]
        motion = motion.transpose(2, 0, 1)[:motion_length]
        motion = motion.reshape(motion_length, -1, 3)
        return motion
