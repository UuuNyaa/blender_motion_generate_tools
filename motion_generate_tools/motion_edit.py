# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import json
import logging
import os

import numpy as np
import torch
from data_loaders.humanml.utils import paramUtil

from mdm.model import cfg_sampler
from mdm.utils import dist_util, fixseed, model_util
from motion_utils import DATA_PATH, HumanML3DConverter, NumpyJSONEncoder, dotdict

SMPL_DATA_PATH = os.path.join(DATA_PATH, 'body_models', 'smpl')
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, 'SMPL_NEUTRAL.pkl')
JOINT_REGRESSOR_TRAIN_EXTRA_PATH = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
MODEL_PATH = os.path.join(DATA_PATH, 'save', 'humanml_trans_enc_512', 'model000200000.pt')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
DATASET_OPT_PATH = os.path.join(DATASET_PATH, 'humanml_opt.txt')


def main():
    HumanML3DConverter.initialize_motion_process(
        np.load(os.path.join(DATA_PATH, 'dataset', 'joints', f'{paramUtil.t2m_tgt_skel_id}.npy')),
        paramUtil.t2m_raw_offsets,
        paramUtil.t2m_kinematic_chain
    )

    seed = 1
    max_frames = 196
    guidance_param = 2.5
    device = 0
    text_condition = "a person jumps"
    num_samples = 1
    batch_size = 1
    dataset = 'humanml'
    prefix_end = 0.25
    suffix_start = 0.75

    fixseed.fixseed(seed)

    dist_util.setup_dist(device)

    conv = HumanML3DConverter(
        max_frames,
        np.load(os.path.join(DATASET_PATH, 'HumanML3D', 'Mean.npy')),
        np.load(os.path.join(DATASET_PATH, 'HumanML3D', 'Std.npy')),
    )

    motion = np.load(os.path.join(DATA_PATH, 'dataset', 'joints', '010639.npy'))[:, :22]
    motion = conv.uniform_skeleton(motion)
    offsets = conv.get_offsets(motion)
    motion_origin = conv.apply_offsets(motion, offsets)
    input_features, motion_length = conv.to_features(motion_origin)

    logging.info("Creating model and diffusion...")
    model = model_util.MDM(
        smpl_model_path=SMPL_MODEL_PATH,
        joint_regressor_train_extra_path=JOINT_REGRESSOR_TRAIN_EXTRA_PATH,
        **model_util.get_model_args(dotdict({
            'dataset': dataset,
            'latent_dim': 512,
            'layers': 8,
            'cond_mask_prob': 0.1,
            'arch': 'trans_enc',
            'emb_trans_dec': False,
        }), dotdict({'num_actions': 1, }))
    )

    diffusion = model_util.create_gaussian_diffusion(dotdict({
        'noise_schedule': 'cosine',
        'sigma_small': True,
        'lambda_vel': 0.0,
        'lambda_rcxyz': 0.0,
        'lambda_fc': 0.0,
    }))

    logging.info(f"Loading checkpoints from [{MODEL_PATH}]...")
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model_util.load_model_wo_clip(model, state_dict)

    model = cfg_sampler.ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if text_condition == '':
        guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_features.shape[-1]
    inpainting_mask = torch.ones_like(
        input_features,
        dtype=torch.bool,
        device=input_features.device
    )  # True means use gt motion
    inpainting_mask[0, :, :, int(prefix_end * motion_length): int(suffix_start * motion_length)] = False  # do inpainting in those frames

    logging.info('### Start sampling')

    sample_features = diffusion.p_sample_loop(
        model,
        (batch_size, model.njoints, model.nfeats, max_frames),
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

    save_motion(conv.unapply_offsets(conv.to_motion(sample_features, motion_length), offsets), 'motion_sample.json')
    save_motion(motion, 'motion_input.json')


def save_motion(motion: np.ndarray, output_json_path: str):
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


if __name__ == "__main__":
    main()
