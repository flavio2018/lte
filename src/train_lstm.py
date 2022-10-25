"""Script to train an LSTM on the LTE task."""


from data.generator import generate_batch, get_vocab_size
from model.lstm import LSTM
from model.test import eval_lstm_padded
from utils.rnn_utils import get_mask, get_hidden_mask, reduce_lens, save_states, populate_first_output, build_first_output, batch_acc
from utils.wandb_utils import log_weights_gradient, log_params_norm
import torch
import hydra
import omegaconf
import wandb


@hydra.main(config_path="../conf/local", config_name="train_lstm")
def train_lstm(cfg):
	print(omegaconf.OmegaConf.to_yaml(cfg))
    
	model = LSTM(
		input_size=get_vocab_size(),
		hidden_size=cfg.hid_size,
		output_size=get_vocab_size(),
		batch_size=cfg.bs).to(cfg.device)

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

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
		LEN, NES = torch.randint(1, cfg.max_len+1, (1,)).item(), torch.randint(1, cfg.max_nes+1, (1,)).item()
		padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(length=LEN, nesting=NES, batch_size=cfg.bs)
		padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
		loss_step, acc_step = step(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, opt, cfg.device)
		wandb.log({
				"loss": loss_step,
				"acc": acc_step,
				"update": i_step,
			})
		log_weights_gradient(model, i_step)
		log_params_norm(model, i_step)

		if i_step % 100 == 0:
			n_valid = i_step / 100
			for v_step in range(10):
				LEN, NES = torch.randint(1, cfg.max_len+1, (1,)).item(), torch.randint(1, cfg.max_nes+1, (1,)).item()
				padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(length=LEN, nesting=NES, batch_size=cfg.bs, split='valid')
				padded_samples_batch, padded_targets_batch = padded_samples_batch.to(cfg.device), padded_targets_batch.to(cfg.device)
				loss_valid_step, acc_valid_step = valid_step(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, cfg.device)
				wandb.log({
					"val_loss": loss_valid_step,
					"val_acc": acc_valid_step,
					"val_update": n_valid*10 + v_step,
				})
			eval_lstm_padded(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, cfg.device)


def step(model, sample, target, samples_len, targets_len, loss, opt, device):
	opt.zero_grad()
	model.train()
	outputs = []
	first_output = {}
	h_dict, c_dict = {}, {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	hid_size = model.h_t_1.size(1)

	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		output = model(sample[:, char_pos, :].squeeze(), hidden_mask)
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(model, h_dict, c_dict, samples_len)
		first_output = populate_first_output(output, samples_len, first_output)
	outputs.append(build_first_output(first_output))
	
	model.set_states(h_dict, c_dict)
	
	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		output = model(target[:, char_pos, :].squeeze(), hidden_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)
	
	count_nonzero = 0
	cumulative_loss = 0
	loss_masks = []
	for char_pos, output in enumerate(outputs):
		loss_masks.append(get_mask(targets_len, device))
		targets_len = reduce_lens(targets_len)
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1)) * loss_masks[-1]
		count_nonzero += (char_loss != 0).sum()
		cumulative_loss += torch.sum(char_loss)
	avg_loss = cumulative_loss / count_nonzero
	acc = batch_acc(outputs, target, loss_masks)
	
	avg_loss.backward()
	opt.step()

	model.detach_states()
	return avg_loss.item(), acc.item()


@torch.no_grad()
def valid_step(model, sample, target, samples_len, targets_len, loss, device):
	model.eval()
	outputs = []
	first_output = {}
	h_dict, c_dict = {}, {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	hid_size = model.h_t_1.size(1)

	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		output = model(sample[:, char_pos, :].squeeze(), hidden_mask)
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(model, h_dict, c_dict, samples_len)
		first_output = populate_first_output(output, samples_len, first_output)
	outputs.append(build_first_output(first_output))
	
	model.set_states(h_dict, c_dict)
	
	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		output = model(target[:, char_pos, :].squeeze(), hidden_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)
	
	count_nonzero = 0
	cumulative_loss = 0
	loss_masks = []
	for char_pos, output in enumerate(outputs):
		loss_masks.append(get_mask(targets_len, device))
		targets_len = reduce_lens(targets_len)
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1)) * loss_masks[-1]
		count_nonzero += (char_loss != 0).sum()
		cumulative_loss += torch.sum(char_loss)
	avg_loss = cumulative_loss / count_nonzero
	acc = batch_acc(outputs, target, loss_masks)
	
	model.detach_states()
	return avg_loss.item(), acc.item()


if __name__ == "__main__":
	train_lstm()
