import torch
from data.generator import get_pos2token, get_target_pos2token, get_vocab_size, get_target_vocab_size
from utils.rnn_utils import save_states, save_states_dntm, get_hidden_mask, get_reading_mask, reduce_lens, populate_first_output, build_first_output, batch_acc
from pdb import set_trace
from collections import OrderedDict


def input_tensors_to_str(x_t):
	pos2token = get_pos2token()
	idx_outputs = [torch.argmax(o).item() for o in x_t]
	return ''.join([pos2token[idx] for idx in idx_outputs])


def target_tensors_to_str(y_t):
	pos2token = get_target_pos2token()
	idx_outputs = [torch.argmax(o).item() for o in y_t]
	return ''.join([pos2token[idx] for idx in idx_outputs])


@torch.no_grad()
def lstm_fwd_padded_batch(model, sample, target, samples_len, targets_len, device):
	model.eval()
	outputs = []
	h_dict, c_dict = {1: {}, 2: {}}, {1: {}, 2: {}}
	first_output = {}
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
	return outputs


@torch.no_grad()
def dntm_fwd_padded_batch(model, sample, target, samples_len, targets_len, device):
	model.eval()
	outputs = []
	h_dict = {}
	first_output = {}
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
	return outputs


@torch.no_grad()
def encdec_fwd_padded_batch(encoder, decoder, sample, target, samples_len, targets_len, device):
	outputs = []
	h_dict, c_dict = {1: {}, 2: {}}, {1: {}, 2: {}}
	samples_len = samples_len.copy()
	targets_len = targets_len.copy()
	hid_size = encoder.h_t_1.size(1)
	
	for char_pos in range(sample.size(1)):
		hidden_mask = get_hidden_mask(samples_len, hid_size, device)
		output = encoder(sample[:, char_pos, :].squeeze(), hidden_mask)
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(encoder, h_dict, c_dict, samples_len)
		
	decoder.set_states(h_dict, c_dict)
	output = decoder(torch.ones(sample[:, char_pos, :].squeeze().size(), device=device),
					 torch.ones(hidden_mask.size(), device=device))
	outputs.append(output)
	targets_len_copy = targets_len.copy()
	targets_len_copy = reduce_lens(targets_len_copy)

	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		output = decoder(target[:, char_pos, :].squeeze(), hidden_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)
	encoder.detach_states()
	decoder.detach_states()
	return outputs


def encdec_dntm_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, device):
	outputs = []
	h_dict = {}
	samples_len = samples_len.copy()

	for char_pos in range(sample.size(1)):
		output = encoder(sample[:, char_pos, :].squeeze())
		samples_len = reduce_lens(samples_len)
		h_dict = save_states_dntm(encoder, h_dict, samples_len)

	decoder.set_states(h_dict)

	for char_pos in range(target.size(1) - 1):
		if outputs:
			output = decoder(outputs[-1])  # no TF
		else:
			output = decoder(target[:, char_pos, :].squeeze())
		outputs.append(output)
	outputs = [final_mlp(o) for o in outputs]
	return outputs


def encdec_step(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, device):
	outputs = []
	h_dict, c_dict = {1: {}, 2: {}}, {1: {}, 2: {}}
	samples_len = samples_len.copy()

	for char_pos in range(sample.size(1)):
		output = encoder(sample[:, char_pos, :].squeeze())
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(encoder, h_dict, c_dict, samples_len)

	decoder.set_states(h_dict, c_dict)

	for char_pos in range(target.size(1) - 1):
		if outputs:
			output = decoder(outputs[-1])  # no teacher forcing
		else:
			output = decoder(target[:, char_pos, :].squeeze())
		outputs.append(output)
	outputs = [final_mlp(o) for o in outputs]
	return outputs


def eval_padded(outputs, target, sample):
	BS = target.size(0)

	samples_str = [input_tensors_to_str([sample[BATCH, p, :] for p in range(sample.size(1))]) for BATCH in range(BS)]	
	targets_str = [target_tensors_to_str([target[BATCH, p, :] for p in range(target.size(1))]) for BATCH in range(BS)]
	outputs_str = [target_tensors_to_str([o[BATCH, :] for o in outputs]) for BATCH in range(BS)]
	
	idx = torch.randint(BS, (1,)).item()
	print(samples_str[idx])
	print("out:", outputs_str[idx])
	print("target:", targets_str[idx])
	print()


def eval_lstm_padded(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, device):
	outputs = lstm_fwd_padded_batch(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	print(compute_loss(loss, outputs, padded_targets_batch).item())
	print(batch_acc(outputs, padded_targets_batch[:, 1:, :], get_vocab_size()).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch)


def eval_dntm_padded(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, device):
	outputs = dntm_fwd_padded_batch(model, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	print(compute_loss(loss, outputs, padded_targets_batch).item())
	print(batch_acc(outputs, padded_targets_batch, get_vocab_size()).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch)

def eval_encdec_padded(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, device):
	with torch.no_grad():
		outputs = encdec_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	encoder.detach_states()
	decoder.detach_states()
	print(compute_loss(loss, outputs, padded_targets_batch[:, 1:, :]).item())
	print(batch_acc(outputs, padded_targets_batch[:, 1:, :], get_target_vocab_size()).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch)

def eval_encdec_dntm_padded(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, device):
	with torch.no_grad():
		outputs = encdec_dntm_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	encoder.detach_states()
	decoder.detach_states()
	print(compute_loss(loss, outputs, padded_targets_batch[:, 1:, :]).item())
	print(batch_acc(outputs, padded_targets_batch[:, 1:, :], get_target_vocab_size()).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch)

def compute_loss(loss, outputs, target):
	# for b in range(32):
	# 	print(f"{b} output: {''.join([target_tensors_to_str([o[b, :] for o in outputs])])}")
	# 	print(f"{b} target: {target_tensors_to_str(target[b, :, :])}")
	# print("-")
	cumulative_loss = 0
	idx_pad = get_target_vocab_size() - 1
	idx_targets = target.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	for char_pos, output in enumerate(outputs):
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1))
		masked_char_loss = char_loss * mask[:, char_pos]
		cumulative_loss += masked_char_loss.sum()
		# print("avg std logits: {:3f}".format(output.std(1).mean().cpu().detach().numpy()))
		# print(f"nonzero: {torch.count_nonzero(masked_char_loss)}")
		# print("cum loss: {:3f}".format(cumulative_loss.item()))
		# print("-")
	avg_loss = cumulative_loss / mask.sum()
	# _, _, _, avg_len_diff = get_num_unequal(outputs, target)  # penalize outputs of different length than target
	# avg_loss += avg_len_diff
	# print("avg loss: {:3f}".format(avg_loss.item()))
	# print()
	return avg_loss


def get_lengths(batch):
	EOS_idx = batch.size(2) - 2
	lengths_dict = OrderedDict()  # seq -> len
	# for each sequence, get positions where idx == EOS 
	for s, l in torch.argwhere((batch.argmax(2) == EOS_idx).type(torch.int)):
		lengths_dict.setdefault(s.item(), l.item())  # get only the first occurence for each seq (i.e., seq length)
	return lengths_dict


def get_num_unequal(x, y):
	def fill_missing(lengths):
		for i in range(y.size(0)):
			if i not in lengths:
				lengths[i] = 0
		return lengths

	x = torch.concat([o.unsqueeze(1) for o in x], dim=1)
	lengths_x = get_lengths(x)
	lengths_y = get_lengths(y)
	lengths_x = fill_missing(lengths_x)

	lengths_x = torch.Tensor(list(lengths_x.values()))
	lengths_y = torch.Tensor(list(lengths_y.values()))

	num_unequal = 0
	num_longer = 0
	num_shorter = 0
	avg_len_diff = 0
	
	num_unequal = (lengths_x != lengths_y).sum().item()
	num_longer = (lengths_x > lengths_y).sum().item()
	num_shorter = (lengths_x < lengths_y).sum().item()
	avg_len_diff = (abs(lengths_x - lengths_y)).sum().item()

	return (num_unequal, num_longer, num_shorter, avg_len_diff/len(lengths_x))