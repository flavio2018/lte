"""Script to train a D-NTM  on the LTE task."""


from data.generator import generate_batch, get_vocab_size
from model.dntm.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from model.dntm.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory
from utils.rnn_utils import get_mask, get_hidden_mask, get_reading_mask, reduce_lens, save_states_dntm, populate_first_output, build_first_output, batch_acc
from utils.wandb_utils import log_weights_gradient, log_params_norm
import torch
import wandb


MAX_ITER = 30000
N_LOCATIONS = 500
CONTENT_SIZE = 8
ADDRESS_SIZE = 8
HID_SIZE = 100
BS = 64
LR = 0.001
LEN = 1
NES = 1
DEVICE = 'cuda'


def train_dntm():
	dntm_memory = DynamicNeuralTuringMachineMemory(
		n_locations=N_LOCATIONS,
		content_size=CONTENT_SIZE,
		address_size=ADDRESS_SIZE,
		controller_input_size=get_vocab_size(),
		controller_hidden_state_size=HID_SIZE)
	model = DynamicNeuralTuringMachine(
		memory=dntm_memory,
		controller_hidden_state_size=100,
		controller_input_size=get_vocab_size(),
		controller_output_size=get_vocab_size()).to(DEVICE)

	loss = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(model.parameters(), lr=LR)

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = "dntm"

	print("MAX_ITER:", MAX_ITER)
	print("hidden_size:", HID_SIZE)
	print("lr:", LR)
	print("optim:", "Adam")
	print("length:", LEN)
	print("nesting:", NES)
	print("batch_size:", BS)
	print("device:", DEVICE)
	print("n_locations:", N_LOCATIONS)
	print("content_size:", CONTENT_SIZE)
	print("address_size:", ADDRESS_SIZE)


	for i_step in range(MAX_ITER):
		batched_samples, batched_targets, samples_len, targets_len = generate_batch(length=LEN, nesting=NES, batch_size=BS)
		batched_samples, batched_targets = batched_samples.to(DEVICE), batched_targets.to(DEVICE)
		loss_step, acc_step = step(model, batched_samples, batched_targets, samples_len, targets_len, loss, opt, DEVICE)
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
				padded_samples_batch, padded_targets_batch, samples_len, targets_len = generate_batch(length=LEN, nesting=NES, batch_size=BS, split='valid')
				padded_samples_batch, padded_targets_batch = padded_samples_batch.to(DEVICE), padded_targets_batch.to(DEVICE)
				loss_valid_step, acc_valid_step = valid_step(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, DEVICE)
				wandb.log({
					"val_loss": loss_valid_step,
					"val_acc": acc_valid_step,
					"val_update": n_valid*10 + v_step,
				})


def step(model, sample, target, samples_len, targets_len, loss, opt, device):
	opt.zero_grad()
	model.train()
	outputs = []
	first_output = {}
	h_dict = {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	feature_size = sample.size(2)
	batch_size = sample.size(0)
	
	model.prepare_for_batch(sample, sample.device)
	hid_size = model.controller_hidden_state.size(0)
	mem_size = model.memory.overall_memory_size
	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		reading_mask = get_reading_mask(samples_len, mem_size, device)
		output = model(sample[:, char_pos, :].reshape(feature_size, batch_size), hidden_mask, reading_mask)
		samples_len = reduce_lens(samples_len)
		h_dict = save_states_dntm(model, h_dict, samples_len)
		first_output = populate_first_output(output, samples_len, first_output)
	outputs.append(build_first_output(first_output))
	
	model.set_states(h_dict)

	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		reading_mask = get_reading_mask(samples_len, mem_size, device)
		output = model(target[:, char_pos, :].reshape(feature_size, batch_size), hidden_mask, reading_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)

	count_nonzero = 0
	cumulative_loss = 0
	loss_masks = []
	for char_pos, output in enumerate(outputs):
		loss_masks.append(get_mask(targets_len, device))
		targets_len = reduce_lens(targets_len)
		char_loss = loss(output.squeeze(), torch.argmax(target[:, char_pos, :].squeeze(), dim=1)) * loss_masks[-1]
		count_nonzero += (char_loss != 0).sum()
		cumulative_loss += torch.sum(char_loss)
	avg_loss = cumulative_loss / count_nonzero
	acc = batch_acc(outputs, target, loss_masks)

	avg_loss.backward()
	opt.step()

	return avg_loss.item(), acc.item()


@torch.no_grad()
def valid_step(model, sample, target, samples_len, targets_len, loss, device):
	model.eval()
	outputs = []
	first_output = {}
	h_dict = {}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	feature_size = sample.size(2)
	batch_size = sample.size(0)
	
	model.prepare_for_batch(sample, sample.device)
	hid_size = model.controller_hidden_state.size(0)
	mem_size = model.memory.overall_memory_size
	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		reading_mask = get_reading_mask(samples_len, mem_size, device)
		output = model(sample[:, char_pos, :].reshape(feature_size, batch_size), hidden_mask, reading_mask)
		samples_len = reduce_lens(samples_len)
		h_dict = save_states_dntm(model, h_dict, samples_len)
		first_output = populate_first_output(output, samples_len, first_output)
	outputs.append(build_first_output(first_output))
	
	model.set_states(h_dict)

	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		reading_mask = get_reading_mask(samples_len, mem_size, device)
		output = model(target[:, char_pos, :].reshape(feature_size, batch_size), hidden_mask, reading_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)

	count_nonzero = 0
	cumulative_loss = 0
	loss_masks = []
	for char_pos, output in enumerate(outputs):
		loss_masks.append(get_mask(targets_len, device))
		targets_len = reduce_lens(targets_len)
		char_loss = loss(output.squeeze(), torch.argmax(target[:, char_pos, :].squeeze(), dim=1)) * loss_masks[-1]
		count_nonzero += (char_loss != 0).sum()
		cumulative_loss += torch.sum(char_loss)
	avg_loss = cumulative_loss / count_nonzero
	acc = batch_acc(outputs, target, loss_masks)

	return avg_loss.item(), acc.item()


if __name__ == "__main__":
	train_dntm()
