#! /bin/bash

python train_encdec.py codename=encdec_sos max_len=1 max_nes=1
python train_encdec.py codename=encdec_sos max_len=1 max_nes=2
python train_encdec.py codename=encdec_sos max_len=2 max_nes=1