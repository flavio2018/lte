#! /bin/bash

python src/train_dntm.py codename=dntm_literal_len max_iter=3000 bs=128 lr=1e-3 max_len=1 max_nes=1
python src/train_dntm.py codename=dntm_literal_len max_iter=3000 bs=128 lr=1e-3 max_len=1 max_nes=2
python src/train_dntm.py codename=dntm_literal_len max_iter=3000 bs=128 lr=1e-3 max_len=2 max_nes=1
