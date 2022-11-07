import torch
from data.generator import get_pos2token, get_target_pos2token, get_vocab_size, get_target_vocab_size
from utils.rnn_utils import save_states, save_states_dntm, get_hidden_mask, get_reading_mask, reduce_lens, populate_first_output, build_first_output, batch_acc


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


def encdec_step(encoder, decoder, sample, target, samples_len, targets_len, device):
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

	targets_len_copy = targets_len.copy()
	for char_pos in range(target.size(1) - 1):
		hidden_mask = get_hidden_mask(targets_len_copy, hid_size, device)
		output = decoder(target[:, char_pos, :].squeeze(), hidden_mask)
		targets_len_copy = reduce_lens(targets_len_copy)
		outputs.append(output)
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

def eval_encdec_padded(encoder, decoder, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, device):
    with torch.no_grad():
        outputs = encdec_step(encoder, decoder, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
    encoder.detach_states()
    decoder.detach_states()
    print(compute_loss(loss, outputs, padded_targets_batch[:, 1:, :]).item())
    print(batch_acc(outputs, padded_targets_batch[:, 1:, :], get_target_vocab_size()).item())
    eval_padded(outputs, padded_targets_batch, padded_samples_batch)

def compute_loss(loss, outputs, target):
	cumulative_loss = 0
	idx_pad = get_target_vocab_size() - 1
	idx_targets = target.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	for char_pos, output in enumerate(outputs):
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1))
		masked_char_loss = char_loss * mask[:, char_pos]
		cumulative_loss += masked_char_loss.sum()
		# set_trace()
	avg_loss = cumulative_loss / mask.sum()
	return avg_loss
