#! /bin/bash

python src/train_lstm.py codename=debug_homog_batch max_nes=1 max_len=3 max_iter=100000 bs=2 lr=3e-4
python src/train_lstm.py codename=debug_homog_batch max_nes=1 max_len=4 max_iter=100000 bs=2 lr=3e-4
python src/train_lstm.py codename=debug_homog_batch max_nes=1 max_len=5 max_iter=100000 bs=2 lr=3e-4
python src/train_lstm.py codename=debug_homog_batch max_nes=2 max_len=1 max_iter=100000 bs=2 lr=3e-4
python src/train_lstm.py codename=debug_homog_batch max_nes=2 max_len=2 max_iter=100000 bs=2 lr=3e-4
python src/train_lstm.py codename=debug_homog_batch max_nes=2 max_len=3 max_iter=100000 bs=2 lr=3e-4