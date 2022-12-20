# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

# This code is based on https://github.com/GuyTevet/motion-diffusion-model/blob/main/sample/edit.py

import json
import logging
import os
from typing import List

import numpy as np
import torch

from data_loaders.humanml.utils import paramUtil
from filepaths import (CLIP_DATA_PATH, DATASET_PATH,
                       JOINT_REGRESSOR_TRAIN_EXTRA_PATH, MDM_MODEL_PATH,
                       SMPL_MODEL_PATH)
from model import cfg_sampler
from motion_utils import HumanML3DConverter, NumpyJSONEncoder, dotdict
from utils import dist_util, fixseed, model_util

MAX_FRAMES = 196


class InBetweenMotionGenerator:
    def __init__(self, device: int = 0):
        HumanML3DConverter.initialize_motion_process(
            np.load(os.path.join(DATASET_PATH, 'joints', f'{paramUtil.t2m_tgt_skel_id}.npy')),
            paramUtil.t2m_raw_offsets,
            paramUtil.t2m_kinematic_chain
        )

        dist_util.setup_dist(device)

        self.converter = HumanML3DConverter(
            MAX_FRAMES,
            np.load(os.path.join(DATASET_PATH, 'HumanML3D', 'Mean.npy')),
            np.load(os.path.join(DATASET_PATH, 'HumanML3D', 'Std.npy')),
        )

        dataset = 'humanml'

        logging.info("Creating model and diffusion...")
        model = model_util.MDM(
            smpl_model_path=SMPL_MODEL_PATH,
            joint_regressor_train_extra_path=JOINT_REGRESSOR_TRAIN_EXTRA_PATH,
            clip_download_root=CLIP_DATA_PATH, device=None if torch.cuda.is_available() else 'cpu',
            **model_util.get_model_args(dotdict({
                'dataset': dataset,
                'latent_dim': 512,
                'layers': 8,
                'cond_mask_prob': 0.1,
                'arch': 'trans_enc',
                'emb_trans_dec': False,
            }), dotdict({'num_actions': 1, }))
        )

        logging.info(f"Loading checkpoints from [{MDM_MODEL_PATH}]...")
        state_dict = torch.load(MDM_MODEL_PATH, map_location='cpu')
        model_util.load_model_wo_clip(model, state_dict)

        model = cfg_sampler.ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking

        self.model = model

    def generate(self, text_condition: str, diffusion_sampling_steps: int) -> List:
        seed = 1
        guidance_param = 2.5
        num_samples = 1
        batch_size = 1
        prefix_end = 0.25
        suffix_start = 0.75

        diffusion_steps = 1000

        self.diffusion = model_util.create_gaussian_diffusion(dotdict({
            'diffusion_steps': diffusion_steps,
            'diffusion_sampling_steps': diffusion_sampling_steps,
            'noise_schedule': 'cosine',
            'sigma_small': True,
            'lambda_vel': 0.0,
            'lambda_rcxyz': 0.0,
            'lambda_fc': 0.0,
        }))

        fixseed.fixseed(seed)

        motion = np.load(os.path.join(DATASET_PATH, 'joints', '000021.npy'))[:, :22]
        motion = self.converter.uniform_skeleton(motion)  # scale 1.109527
        offsets = self.converter.get_offsets(motion)
        motion_origin = self.converter.apply_offsets(motion, offsets)
        input_features, motion_length = self.converter.to_features(motion_origin)

        if text_condition == '':
            guidance_param = 0.  # Force unconditioned generation

        # add inpainting mask according to args
        assert MAX_FRAMES == input_features.shape[-1]
        inpainting_mask = torch.ones_like(
            input_features,
            dtype=torch.bool,
            device=input_features.device
        )  # True means use gt motion
        inpainting_mask[0, :, :, int(prefix_end * motion_length): int(suffix_start * motion_length)] = False  # do inpainting in those frames

        logging.info('### Start sampling')

        sample_features = self.diffusion.p_sample_loop(
            self.model,
            (batch_size, self.model.njoints, self.model.nfeats, MAX_FRAMES),
            clip_denoised=False,
            model_kwargs={'y': {
                'text': [text_condition] * num_samples,
                'inpainted_motion': input_features,
                'inpainting_mask': inpainting_mask,
                'scale': torch.ones(batch_size, device=dist_util.dev()) * guidance_param,  # add CFG scale to batch
            }},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample_motion = self.converter.unapply_offsets(self.converter.to_motion(sample_features, motion_length), offsets)
        sample_position_quaternion_motion = self.converter.to_pq_motion(sample_features, motion_length, offsets)
        return sample_position_quaternion_motion.tolist()

    def save_as_json(self, array: np.ndarray, output_json_path: str):
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(array.tolist(), outfile, indent=2, cls=NumpyJSONEncoder)

    def save_motion(self, motion: np.ndarray, output_json_path: str):
        kinematic_tree = paramUtil.t2m_kinematic_chain
        chain_names = ['leg.r', 'leg.l', 'spine', 'arm.r', 'arm.l']

        frame_count = motion.shape[0]

        chain_frames = [
            {
                chain_name: list(zip(
                    motion[frame, chain, 0],
                    motion[frame, chain, 1],
                    motion[frame, chain, 2]
                ))
                for chain_name, chain in zip(chain_names, kinematic_tree)
            }
            for frame in range(frame_count)
        ]

        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(chain_frames, outfile, indent=2, cls=NumpyJSONEncoder)
