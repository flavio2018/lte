#! /bin/bash
# train deep lstm on max_len=4, max_nes=1 with 4 different hid layer sizes
# and default hyperparams

python src/train_lstm.py codename=deep_lstm hid_size=256
python src/train_lstm.py codename=deep_lstm hid_size=512
python src/train_lstm.py codename=deep_lstm hid_size=1024
python src/train_lstm.py codename=deep_lstm hid_size=2048