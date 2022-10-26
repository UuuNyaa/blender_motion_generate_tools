#!/bin/sh

# install dependent modules
pip3 --disable-pip-version-check --no-cache-dir install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

# TODO download and place data