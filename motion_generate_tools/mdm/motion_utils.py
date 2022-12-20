# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders import tensors
from data_loaders.humanml.common import quaternion, skeleton
from data_loaders.humanml.scripts import motion_process
from utils import dist_util


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


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


@dataclass
class MotionOffsets:
    offset_position: np.ndarray
    offset_quaternion: np.ndarray


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

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).

        Returns:
            uniformed_motion: An uniformed joint position motion as ndarray of shape (frame, 22, 3).
        """
        return motion_process.uniform_skeleton(motion, motion_process.tgt_offsets)

    def get_offsets(self, motion: np.ndarray) -> MotionOffsets:
        """
        Get offsets.

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).

        Returns:
            motion_offsets: MotionOffsets
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

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).
            offsets: MotionOffsets

        Returns:
            offseted_motion: Offsetted joint position motion as ndarray of shape (frame, 22, 3).
        """
        motion = motion - offsets.offset_position
        motion = quaternion.qrot_np(np.ones(motion.shape[:-1] + (4,)) * offsets.offset_quaternion, motion)
        return motion

    def unapply_offsets(self, motion: np.ndarray, offsets: MotionOffsets) -> np.ndarray:
        """
        Unapply offsets.

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).
            offsets: MotionOffsets

        Returns:
            unoffseted_motion: Unoffsetted joint position motion as ndarray of shape (frame, 22, 3).
        """
        motion = quaternion.qrot_np(np.ones(motion.shape[:-1] + (4,)) * quaternion.qinv_np(offsets.offset_quaternion), motion)
        motion = motion + offsets.offset_position
        return motion

    def to_features(self, motion: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        Convert from joint position motion to features.

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).

        Returns:
            A tuple (features, motion_length), where features as tensor of shape (1, 263, 1, frame) and motion_length as int.
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
        Convert from features to joint position motion.

        Args:
            features: A features as tensor of shape (1, 263, 1, frame)

        Returns:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).
        """

        features = ((features.cpu().permute(0, 2, 3, 1)) * self.std + self.mean).float()
        features = motion_process.recover_from_ric(features, motion_process.joints_num)
        features = features.view(-1, *features.shape[2:]).permute(0, 2, 3, 1)
        motion: np.ndarray = features.cpu().numpy()[0]
        motion = motion.transpose(2, 0, 1)[:motion_length]
        motion = motion.reshape(motion_length, -1, 3)
        return motion

    def to_joint_rotation_motion(self, features: torch.Tensor, motion_length: int) -> np.ndarray:
        """
        Convert from features to joint rotation motion.

        Args:
            features: A features as tensor of shape (1, 263, 1, frame)

        Returns:
            motion: A joint rotation motion as ndarray of shape (frame, 22, 6).
        """

        features = ((features.cpu().permute(0, 2, 3, 1)) * self.std + self.mean).float()

        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(features)
        r_rot_cont6d = quaternion.quaternion_to_cont6d(r_rot_quat)
        start_indx = 1 + 2 + 1 + (motion_process.joints_num - 1) * 3
        end_indx = start_indx + (motion_process.joints_num - 1) * 6
        cont6d_params = features[..., start_indx:end_indx]
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, motion_process.joints_num, 6)
        matrices = quaternion.cont6d_to_matrix(cont6d_params[:motion_length])
        quaternions = matrix_to_quaternion(matrices)
        return quaternions.numpy()

    def to_pq_motion(self, features: torch.Tensor, motion_length: int, offsets: MotionOffsets) -> np.ndarray:
        features = ((features.cpu().permute(0, 2, 3, 1)) * self.std + self.mean).float()
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(features)
        positions = features[..., 4:(motion_process.joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = quaternion.qrot(quaternion.qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        positions = positions.view(-1, motion_process.joints_num, 3)[:motion_length]

        # apply offsets
        offset_quaternion = quaternion.qinv(torch.from_numpy(offsets.offset_quaternion))
        positions = quaternion.qrot(torch.ones(positions.shape[:-1] + (4,)) * offset_quaternion, positions)
        positions = positions + torch.from_numpy(offsets.offset_position)

        r_rot_cont6d = quaternion.quaternion_to_cont6d(quaternion.qmul(r_rot_quat, offset_quaternion.repeat(r_rot_quat.shape[:-1] + (1,))))
        start_indx = 1 + 2 + 1 + (motion_process.joints_num - 1) * 3
        end_indx = start_indx + (motion_process.joints_num - 1) * 6
        cont6d = features[..., start_indx:end_indx]
        cont6d = torch.cat([r_rot_cont6d, cont6d], dim=-1)
        cont6d = cont6d.view(-1, motion_process.joints_num, 6)
        matrices = quaternion.cont6d_to_matrix(cont6d)
        quaternions = matrix_to_quaternion(matrices)[:motion_length]

        return torch.cat([positions, quaternions], dim=2).numpy()

    def position_motion_to_quaternion_motion(self, motion: np.ndarray) -> np.ndarray:
        """
        Convert from joint position motion to joint rotation motion.

        Args:
            motion: A joint position motion as ndarray of shape (frame, 22, 3).

        Returns:
            motion: A joint quaternion motion as ndarray of shape (frame, 22, 4).
        """
        skel = skeleton.Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, 'cpu')
        quat_params = skel.inverse_kinematics_np(motion, motion_process.face_joint_indx, smooth_forward=True)
        return quat_params
