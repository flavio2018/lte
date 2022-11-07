#! /bin/bash

python train_encdec.py codename=encdec_if max_len=1 max_nes=1 ops=i
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=2 ops=as hid_size=2048
python train_encdec.py codename=encdec_eos_sum_sub max_len=1 max_nes=3 ops=as hid_size=2048
python train_encdec.py codename=encdec_2048 max_len=1 max_nes=1 ops=ai hid_size=2048
python train_encdec.py codename=encdec_2048 max_len=2 max_nes=1 ops=a hid_size=2048
