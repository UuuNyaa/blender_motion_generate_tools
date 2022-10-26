import logging
import os
import sys

import torch

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d | %(levelname).4s | %(module)s | %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

WORK_PATH = os.path.dirname(__file__)
MDM_PATH = os.path.abspath(os.path.join(WORK_PATH, 'mdm'))
DATA_PATH = os.path.abspath(os.path.join(WORK_PATH, '..', 'data'))

sys.path.append(MDM_PATH)

from mdm.data_loaders import get_data, tensors
from mdm.data_loaders.humanml.scripts import motion_process
from mdm.data_loaders.humanml.utils import paramUtil
from mdm.model import cfg_sampler
from mdm.utils import dist_util, fixseed, model_util

ENC_MODEL_PATH = os.path.join(DATA_PATH, 'save/humanml_trans_enc_512/model000200000.pt')


def disable_tqdm_globally():
    from functools import partialmethod

    from tqdm import tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    disable_tqdm_globally()

    seed = 1
    motion_length = 2.0  # secs
    max_frames = 196
    fps = 30
    n_frames = min(max_frames, int(motion_length * fps))
    guidance_param = 2.5
    device = 0
    text_prompt = "a person jumps"
    num_repetitions = 1
    batch_size = 1

    fixseed.fixseed(seed)

    dist_util.setup_dist(device)

    model = model_util.MDM(**model_util.get_model_args(dotdict({
        'dataset': 'humanml',
        'smpl_data_path': os.path.join(DATA_PATH, 'body_models/smpl'),
        'latent_dim': 512,
        'layers': 8,
        'cond_mask_prob': 0.1,
        'arch': 'trans_enc',
        'emb_trans_dec': False,
    })))

    diffusion = model_util.create_gaussian_diffusion(dotdict({
        'noise_schedule': 'cosine',
        'sigma_small': True,
        'lambda_vel': 0.0,
        'lambda_rcxyz': 0.0,
        'lambda_fc': 0.0,
    }))

    state_dict = torch.load(ENC_MODEL_PATH, map_location='cpu')
    model_util.load_model_wo_clip(model, state_dict)

    model = cfg_sampler.ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    data = get_data.get_dataset(
        name='humanml',
        num_frames=max_frames,
        datapath=os.path.join(DATA_PATH, 'humanml_opt.txt'),
        split='test',
        hml_mode='text_only',
    )

    _, model_kwargs = tensors.collate([{
        'inp': torch.tensor([[0.]]),
        'target': 0,
        'text': text_prompt,
        'tokens': None,
        'lengths': n_frames
    }])

    all_text = []
    all_motions = []
    all_lengths = []

    for rep_i in range(num_repetitions):
        logging.info(f'### Start sampling [repetitions #{rep_i}]')

        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * guidance_param

        sample = diffusion.p_sample_loop(
            model,
            (batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = motion_process.recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        logging.info(f"created {len(all_motions) * batch_size} samples")

    skeleton = paramUtil.t2m_kinematic_chain


if __name__ == '__main__':
    main()
