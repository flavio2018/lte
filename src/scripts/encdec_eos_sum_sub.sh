#! /bin/bash

python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=1
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=2
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=3
python train_encdec.py codename=encdec_eos_sum_sub max_len=2 max_nes=1 hid_size=1024
python train_encdec.py codename=encdec_eos_sum_sub max_len=2 max_nes=2 hid_size=1024
python train_encdec.py codename=encdec_eos_sum_sub max_len=2 max_nes=3 hid_size=1024