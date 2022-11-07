#! /bin/bash

python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=1
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=2
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=3