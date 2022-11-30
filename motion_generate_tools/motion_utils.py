import json
import logging
import os

import numpy as np

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
