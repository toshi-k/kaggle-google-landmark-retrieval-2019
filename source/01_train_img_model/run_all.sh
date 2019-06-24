#!/bin/bash

project=img_model

python 01_train_embedding.py -n ${project}

python 02_valid_model.py -n ${project}

python 03_embed_index.py -n ${project}

python 04_embed_test.py -n ${project}

python 05_make_submission.py -n ${project}
