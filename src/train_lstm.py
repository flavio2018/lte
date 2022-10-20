"""Script to train an LSTM on the LTE task."""


from data.generator import generate_batch, get_vocab_size
from model.lstm import LSTM
from utils.rnn_utils import get_mask, get_hidden_mask, reduce_lens, save_states, populate_first_output, build_first_output
from utils.wandb_utils import log_weights_gradient, log_params_norm
import torch
import wandb


MAX_ITER = 10000
BS = 16
LR = 0.001
LEN = 2
NES = 1
HID_SIZE = 1024
DEVICE = 'cuda'


def train_lstm():
	model = LSTM(
		input_size=get_vocab_size(),
		hidden_size=HID_SIZE,
		output_size=get_vocab_size()).to(DEVICE)

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(model.parameters(), lr=LR)

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = "lstm"


	print("MAX_ITER:", MAX_ITER)
	print("hidden_size:", HID_SIZE)
	print("lr:", LR)
	print("optim:", "Adam")
	print("length:", LEN)
	print("nesting:", NES)
	print("batch_size:", BS)
	print("device:", DEVICE)

	
	for i_step in range(MAX_ITER):
		padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(length=LEN, nesting=NES, batch_size=BS)
		padded_samples_batch, padded_targets_batch = padded_samples_batch.to(DEVICE), padded_targets_batch.to(DEVICE)
		loss_step = step(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, opt, DEVICE)
		wandb.log({
				"loss": loss_step,
				"update": i_step,
			})
		log_weights_gradient(model, step)
		log_params_norm(model, step)


def step(model, sample, target, samples_len, targets_len, loss, opt, device):
	opt.zero_grad()
	model.train()
	outputs = []
	first_output = {}
	h_dict, c_dict = {}, {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	hid_size = model.h_t.size(1)

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
	for char_pos, output in enumerate(outputs):
		loss_mask = get_mask(targets_len, device)
		targets_len = reduce_lens(targets_len)
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1)) * loss_mask
		count_nonzero += (char_loss != 0).sum()
		cumulative_loss += torch.sum(char_loss)
	avg_loss = cumulative_loss / count_nonzero
	
	avg_loss.backward()
	opt.step()

	model.detach_states()
	return avg_loss.item()


if __name__ == "__main__":
	train_lstm()
