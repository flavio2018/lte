#! /bin/bash

python src/train_lstm.py codename=deep_lstm_add bs=256
python src/train_dntm.py codename=dntm_add bs=256
python src/train_encdec.py codename=deep_lstm_encdec_add bs=256