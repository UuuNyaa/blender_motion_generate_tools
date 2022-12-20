# -*- coding: utf-8 -*-

import os

WORK_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(WORK_PATH, '..', 'data'))

MDM_MODEL_PATH = os.path.join(DATA_PATH, 'save', 'humanml_trans_enc_512', 'model000200000.pt')

CLIP_DATA_PATH = os.path.join(DATA_PATH, 'save', 'clip')
CLIP_MODEL_NAME = 'ViT-B/32'
CLIP_MODEL_PATH = os.path.join(CLIP_DATA_PATH, 'ViT-B-32.pt')

SMPL_DATA_PATH = os.path.join(DATA_PATH, 'body_models', 'smpl')
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, 'SMPL_NEUTRAL.pkl')
JOINT_REGRESSOR_TRAIN_EXTRA_PATH = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
DATASET_OPT_PATH = os.path.join(DATASET_PATH, 'humanml_opt.txt')
