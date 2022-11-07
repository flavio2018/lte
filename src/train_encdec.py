"""Train encoder-decoder models on LTE task."""

from model.lstm import LSTM, DeepLSTM
from model.test import eval_encdec_padded, compute_loss, encdec_step
from data.generator import get_vocab_size, get_target_vocab_size, generate_batch
from utils.rnn_utils import get_mask, get_hidden_mask, reduce_lens, save_states, populate_first_output, build_first_output, batch_acc
from utils.wandb_utils import log_weights_gradient, log_params_norm
import torch
import wandb
import hydra
import omegaconf


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

	enc_dec_parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()]

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(enc_dec_parameters, lr=cfg.lr)

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = cfg.codename
	wandb.config.update(omegaconf.OmegaConf.to_container(
		cfg, resolve=True, throw_on_missing=True))

	for i_step in range(cfg.max_iter):
		#LEN, NES = torch.randint(1, cfg.max_len+1, (1,)).item(), torch.randint(1, cfg.max_nes+1, (1,)).item()
		LEN, NES = cfg.max_len, cfg.max_nes
		padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(LEN, NES, cfg.bs, ops=cfg.ops)
		padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
		loss_step, acc_step = train_step(encoder, decoder, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, opt, cfg.device)
		wandb.log({
				"loss": loss_step,
				"acc": acc_step,
				"update": i_step,
			})
		#log_weights_gradient(encoder, i_step)
		log_weights_gradient(decoder, i_step)
		#log_params_norm(encoder, i_step)
		log_params_norm(decoder, i_step)
		eval_encdec_padded(encoder, decoder, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)

		if i_step % 100 == 0:
			n_valid = i_step / 100
			for v_step in range(10):
				#LEN, NES = torch.randint(1, cfg.max_len+1, (1,)).item(), torch.randint(1, cfg.max_nes+1, (1,)).item()
				LEN, NES = cfg.max_len, cfg.max_nes
				padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(LEN, NES, cfg.bs, split='valid', ops=cfg.ops)
				padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
				loss_valid_step, acc_valid_step = valid_step(encoder, decoder, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)
				wandb.log({
					"val_loss": loss_valid_step,
					"val_acc": acc_valid_step,
					"val_update": n_valid*10 + v_step,
				})


def train_step(encoder, decoder, sample, target, samples_len, targets_len, loss, opt, device):
	opt.zero_grad()
	encoder.train()
	decoder.train()

	outputs = encdec_step(encoder, decoder, sample, target, samples_len, targets_len, device)
	avg_loss = compute_loss(loss, outputs, target[:, 1:, :])
	acc = batch_acc(outputs, target[:, 1:, :], get_target_vocab_size())  # cut SOS

	avg_loss.backward()
	opt.step()

	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item()


@torch.no_grad()
def valid_step(encoder, decoder, sample, target, samples_len, targets_len, loss, device):
	encoder.eval()
	decoder.eval()

	outputs = encdec_step(encoder, decoder, sample, target, samples_len, targets_len, device)
	avg_loss = compute_loss(loss, outputs, target[:, 1:, :])
	acc = batch_acc(outputs, target[:, 1:, :], get_target_vocab_size())
	
	encoder.detach_states()
	decoder.detach_states()
	return avg_loss.item(), acc.item()


if __name__ == '__main__':
	train_encdec()
