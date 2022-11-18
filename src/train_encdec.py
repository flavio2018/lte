"""Train encoder-decoder models on LTE task."""

from model.lstm import LSTM, DeepLSTM
from model.mlp import MLP
from model.test import eval_encdec_padded, compute_loss, encdec_step
from data.generator import get_vocab_size, get_target_vocab_size, generate_batch, SubsetDataset
from utils.rnn_utils import get_mask, get_hidden_mask, reduce_lens, save_states, populate_first_output, build_first_output, batch_acc
from utils.wandb_utils import log_weights_gradient, log_params_norm
import collections
import numpy as np
import torch
import wandb
import hydra
import omegaconf


LOSS_WIN_SIZE = 10
FREQ_EVAL = 10
FREQ_LOSS_RECORDING = 100001


@hydra.main(config_path="../conf/local", config_name="train_encdec")
def train_encdec(cfg):
	print(omegaconf.OmegaConf.to_yaml(cfg))
	
	encoder = DeepLSTM(
		input_size=get_vocab_size(),
		hidden_size=cfg.hid_size,
		output_size=get_vocab_size(),
		batch_size=cfg.bs,
	).to(cfg.device)

	decoder = DeepLSTM(
		input_size=get_target_vocab_size(),
		hidden_size=cfg.hid_size,
		output_size=get_target_vocab_size(),
		batch_size=cfg.bs,
	).to(cfg.device)

	final_mlp = MLP(
		depth=4,
		input_width=get_target_vocab_size(),
		hidden_width=256,
		output_width=get_target_vocab_size(),
	).to(cfg.device)

	enc_dec_parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()] + [p for p in final_mlp.parameters()]

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(enc_dec_parameters, lr=cfg.lr)
	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.9)
	FREQ_LR_DECAY = cfg.max_iter // 100

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = cfg.codename
	wandb.config.update(omegaconf.OmegaConf.to_container(
		cfg, resolve=True, throw_on_missing=True))
	
	losses = collections.deque([], maxlen=LOSS_WIN_SIZE)	
	# LEN, NES, losses = get_len_nes(1, 1, losses, cfg)
	if cfg.cl_mix:
		LEN, NES = cfg.max_len, cfg.max_nes
	else:
		LEN, NES = 1, 1
	subset_dataset = SubsetDataset(max_len=cfg.max_len, max_nes=cfg.max_nes, ops=cfg.ops)

	for i_step in range(cfg.max_iter):
		padded_samples_batch, padded_targets_batch, samples_len, targets_len = subset_dataset.generate_batch(LEN, NES, cfg.bs, ops=cfg.ops, mod=cfg.mod)
		padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
		loss_step, acc_step = train_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, opt, cfg.device)
		wandb.log({
				"lr": lr_scheduler.get_last_lr()[-1],
				"loss": loss_step,
				"acc": acc_step,
				"update": i_step,
			})
		#log_weights_gradient(encoder, i_step)
		log_weights_gradient(decoder, i_step)
		log_weights_gradient(final_mlp, i_step)
		#log_params_norm(encoder, i_step)
		log_params_norm(decoder, i_step)
		log_params_norm(final_mlp, i_step)
		if i_step % 100 == 0:
			eval_encdec_padded(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)

		padded_samples_batch, padded_targets_batch, samples_len, targets_len = subset_dataset.generate_batch(LEN, NES, cfg.bs, split='valid', ops=cfg.ops, mod=cfg.mod)
		padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
		loss_valid_step, acc_valid_step = valid_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)
		if i_step % FREQ_LOSS_RECORDING == 0:
			losses.append(loss_valid_step)
			LEN, NES, losses = get_len_nes(LEN, NES, losses, cfg)

		if acc_valid_step >= 0.9 and not cfg.cl_mix:
			LEN += 1

		wandb.log({
			"val_loss": loss_valid_step,
			"val_acc": acc_valid_step,
			"update": i_step,
		})

		if (i_step % FREQ_LR_DECAY == 0) and (lr_scheduler.get_last_lr()[-1] > 8e-5):
			lr_scheduler.step()
		
		if i_step % FREQ_EVAL == 0:
			padded_samples_batch, padded_targets_batch, samples_len, targets_len = subset_dataset.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, split='test', ops=cfg.ops, mod=cfg.mod)
			padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
			_, acc_test = valid_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)
			wandb.log({
				"acc_test": acc_test,
				"test_update": i_step // FREQ_EVAL,
			})
		


def train_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, loss, opt, device):
	opt.zero_grad()
	encoder.train()
	decoder.train()

	outputs = encdec_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, device)
	avg_loss = compute_loss(loss, outputs, target[:, 1:, :])
	acc = batch_acc(outputs, target[:, 1:, :], get_target_vocab_size())  # cut SOS

	avg_loss.backward()
	opt.step()

	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item()


@torch.no_grad()
def valid_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, loss, device):
	encoder.eval()
	decoder.eval()

	outputs = encdec_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, device)
	avg_loss = compute_loss(loss, outputs, target[:, 1:, :])
	acc = batch_acc(outputs, target[:, 1:, :], get_target_vocab_size())
	
	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item()


def increase_len_nes(losses):
	if len(losses) < LOSS_WIN_SIZE:
		return False

	# Average change in loss normalized by average loss.
	loss_diffs = [pair[0] - pair[1]
				  for pair in zip(list(losses)[1:],
								  list(losses)[:-1])]
	avg_loss_norm = np.mean(loss_diffs) / np.mean(losses)
	if avg_loss_norm < 0.05:
		return True
	return False


def get_len_nes(cur_len, cur_nes, losses, cfg):
	if increase_len_nes(losses):
		losses = collections.deque([], maxlen=LOSS_WIN_SIZE)
		if cur_len < cfg.max_len and cur_nes < cfg.max_nes:
			print(f"Increased len to {cur_len+1}, nes to {cur_nes+1}")
			return cur_len+1, cur_nes+1, losses
	return cur_len, cur_nes, losses


if __name__ == '__main__':
	train_encdec()
