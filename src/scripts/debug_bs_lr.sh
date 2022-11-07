#! /bin/bash
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=8 lr=3e-3
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=32 lr=3e-3
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=128 lr=3e-3
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=8 lr=3e-5
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=32 lr=3e-5
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=128 lr=3e-5
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=8 lr=3e-7
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=32 lr=3e-7
python src/train_lstm.py codename=debug_bs_lr max_len=1 max_iter=3000 bs=128 lr=3e-7