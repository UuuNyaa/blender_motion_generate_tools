import logging
import os

import numpy as np
from torch.utils import data

from mdm.data_loaders.humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset, WordVectorizer
from mdm.data_loaders.humanml.utils import get_opt

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


class HumanML3DDataset(data.Dataset):
    def __init__(self, mode, base_path, dataset_opt_path, split="train", **kwargs):
        self.mode = mode

        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt.get_opt(dataset_opt_path, device)
        opt.meta_dir = os.path.join(base_path, opt.meta_dir)
        opt.motion_dir = os.path.join(base_path, opt.motion_dir)
        opt.text_dir = os.path.join(base_path, opt.text_dir)
        opt.model_dir = os.path.join(base_path, opt.model_dir)
        opt.checkpoints_dir = os.path.join(base_path, opt.checkpoints_dir)
        opt.data_root = os.path.join(base_path, opt.data_root)
        opt.save_root = os.path.join(base_path, opt.save_root)
        self.opt = opt
        logging.info('Loading dataset %s ...', opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(os.path.join(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(os.path.join(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(os.path.join(opt.data_root, 'Mean.npy'))
            self.std = np.load(os.path.join(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(os.path.join(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(os.path.join(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = os.path.join(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(os.path.join(base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
