#! /bin/bash
# Try dntm with different content sizes (8, 16, 32) on max_len=4, max_nes=1
# and default hyperparameters

python src/train_dntm.py codename=dntm_con_size content_size=8
python src/train_dntm.py codename=dntm_con_size content_size=16
python src/train_dntm.py codename=dntm_con_size content_size=32