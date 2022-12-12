"""Train encoder-decoder models on LTE task."""

from model.lstm import LSTM, DeepLSTM
from model.mlp import MLP
from model.test import eval_encdec_padded, compute_loss, batch_acc, encdec_step, encdec_step_test, get_len_stats, get_lengths
from data.generator import LTEGenerator
from utils.wandb_utils import log_weights_gradient, log_params_norm
import numpy as np
from datetime import datetime as dt
import os
import torch
import wandb
import hydra
import omegaconf


FREQ_EVAL = 10


@hydra.main(config_path="../conf/local", config_name="train_encdec")
def train_encdec(cfg):
	print(omegaconf.OmegaConf.to_yaml(cfg))

	lte = LTEGenerator(device=cfg.device)
	
	encoder = DeepLSTM(
		input_size=len(lte.x_vocab),
		hidden_size=[cfg.hid_size, cfg.hid_size2],
		output_size=len(lte.x_vocab),
		batch_size=cfg.bs,
	).to(cfg.device)

	decoder = DeepLSTM(
		input_size=len(lte.y_vocab),
		hidden_size=[cfg.hid_size, cfg.hid_size2],
		output_size=len(lte.y_vocab),
		batch_size=cfg.bs,
	).to(cfg.device)

	final_mlp = MLP(
		depth=4,
		input_width=len(lte.y_vocab),
		hidden_width=256,
		output_width=len(lte.y_vocab),
	).to(cfg.device)

	enc_dec_parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()] + [p for p in final_mlp.parameters()]

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(enc_dec_parameters, lr=cfg.lr)
	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=cfg.gamma)
	FREQ_LR_DECAY = cfg.max_iter // 100
	FREQ_WANDB_LOG = np.ceil(cfg.max_iter / 100000)  # suggested number of datapoints for wandb scalars

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = cfg.codename
	wandb.config.update(omegaconf.OmegaConf.to_container(
		cfg, resolve=True, throw_on_missing=True))
	start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')
	
	for i_step in range(cfg.max_iter):
		X, Y, len_X, len_Y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops)
		loss_step, acc_step = train_step((encoder, decoder, final_mlp), (X, Y, len_X, len_Y), lte, loss, opt, cfg.device)
		
		if i_step % 100 == 0:
			eval_encdec_padded(encoder, decoder, final_mlp, X, Y, len_X, len_Y, loss, lte, cfg.device)

		if i_step % 1000 == 0:
			torch.save({
					'update': i_step,
					'encoder_state_dict': encoder.state_dict(),
					'decoder_state_dict': decoder.state_dict(),
					'mlp_state_dict': final_mlp.state_dict(),
					'opt': opt.state_dict(),
					'loss_train': loss_step,
				}, os.path.join(hydra.utils.get_original_cwd(), f"../models/checkpoints/{start_timestamp}_{cfg.codename}.pth"))

		X, Y, len_X, len_Y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, split='valid', ops=cfg.ops)
		X, Y = X.to(cfg.device), Y.to(cfg.device)
		loss_valid_step, acc_valid_step, _ = valid_step((encoder, decoder, final_mlp), (X, Y, len_X, len_Y), lte, loss, cfg.device)
		
		if i_step % FREQ_WANDB_LOG == 0:
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
			wandb.log({
				"val_loss": loss_valid_step,
				"val_acc": acc_valid_step,
				"update": i_step,
			})

		if (i_step % FREQ_LR_DECAY == 0) and (lr_scheduler.get_last_lr()[-1] > 8e-5):
			lr_scheduler.step()
		
		if i_step % FREQ_EVAL == 0:
			X, Y, len_X, len_Y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, split='test', ops=cfg.ops)
			X, Y = X.to(cfg.device), Y.to(cfg.device)
			_, acc_test, len_stats = valid_step((encoder, decoder, final_mlp), (X, Y, len_X, len_Y), lte, loss, cfg.device)
			num_unequal, num_longer, num_shorter, sum_len_diff = len_stats
			wandb.log({
				"acc_test": acc_test,
				"test_update": i_step // FREQ_EVAL,
				"num_unequal": num_unequal,
				"num_longer": num_longer,
				"num_shorter": num_shorter,
				"sum_len_diff": sum_len_diff,
			})
		


def train_step(model, data, generator, loss, opt, device):
	encoder, decoder, final_mlp = model
	inputs, targets, _, _ = data
	opt.zero_grad()
	encoder.train()
	decoder.train()

	outputs = encdec_step(encoder, decoder, final_mlp, *data, device)
	avg_loss = compute_loss(loss, outputs, targets[:, 1:, :], generator)
	acc = batch_acc(outputs, targets[:, 1:, :], targets.size(-1), generator)  # cut SOS

	avg_loss.backward()
	opt.step()

	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item()


@torch.no_grad()
def valid_step(model, data, generator, loss, device):
	encoder, decoder, final_mlp = model
	inputs, targets, _, _ = data
	encoder.eval()
	decoder.eval()

	outputs = encdec_step(encoder, decoder, final_mlp, *data, device)
	avg_loss = compute_loss(loss, outputs, targets[:, 1:, :], generator)
	acc = batch_acc(outputs, targets[:, 1:, :], targets.size(-1), generator)
	len_stats = get_len_stats(outputs, targets[:, 1:, :])
	
	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item(), len_stats


if __name__ == '__main__':
	train_encdec()
