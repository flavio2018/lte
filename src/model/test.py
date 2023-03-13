import torch
from data.generator import _PAD
from utils.rnn_utils import save_states, save_states_dntm, reduce_lens
import warnings


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


def encdec_step_test(encoder, decoder, final_mlp, sample, target, samples_len, targets_len, device):
	outputs = []
	h_dict, c_dict = {1: {}, 2: {}}, {1: {}, 2: {}}
	samples_len = samples_len.copy()

	for char_pos in range(sample.size(1)):
		output = encoder(sample[:, char_pos, :].squeeze())
		samples_len = reduce_lens(samples_len)
		h_dict, c_dict = save_states(encoder, h_dict, c_dict, samples_len)

	decoder.set_states(h_dict, c_dict)

	counter = 0
	max_counter = 10 + target.size(1)  # we allow 10 steps more than max length in target
	while counter < max_counter:
		counter += 1
		if outputs:
			output = decoder(outputs[-1])  # no teacher forcing
		else:
			output = decoder(target[:, 0, :].squeeze())
		outputs.append(output)
	outputs = [final_mlp(o) for o in outputs]
	return outputs
	

def eval_padded(outputs, target, sample, generator):
	BS = target.size(0)

	outputs = torch.concat([o.unsqueeze(1) for o in outputs], dim=1)

	samples_str = generator.x_to_str(sample)	
	targets_str = generator.y_to_str(target)
	outputs_str = generator.y_to_str(outputs)
	
	idx = torch.randint(BS, (1,)).item()
	print(samples_str[idx])
	print("out:", outputs_str[idx])
	print("target:", targets_str[idx])
	print()


def eval_encdec_padded(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, generator, device):
	with torch.no_grad():
		outputs = encdec_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	encoder.detach_states()
	decoder.detach_states()
	print(compute_loss(loss, outputs, padded_targets_batch[:, 1:, :], generator).item())
	print(batch_acc(outputs, padded_targets_batch[:, 1:, :], len(generator.y_vocab), generator).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch, generator)

def eval_encdec_dntm_padded(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, loss, generator, device):
	with torch.no_grad():
		outputs = encdec_dntm_step(encoder, decoder, final_mlp, padded_samples_batch, padded_targets_batch, samples_len, targets_len, device)
	encoder.detach_states()
	decoder.detach_states()
	print(compute_loss(loss, outputs, padded_targets_batch[:, 1:, :], generator).item())
	print(batch_acc(outputs, padded_targets_batch[:, 1:, :], len(generator.y_vocab), generator).item())
	eval_padded(outputs, padded_targets_batch, padded_samples_batch, generator)

def eval_ut(ut, X, Y, loss, generator, device):
	ut.eval()
	with torch.no_grad():
		outputs, act = ut(X, Y[:, :-1])
	list_outputs = [outputs[:, pos, :] for pos in range(outputs.size(1))]
	print(compute_loss(loss, list_outputs, Y[:, 1:], generator).item())
	print(batch_acc(list_outputs, Y[:, 1:], Y.size(-1), generator).item())
	eval_padded(list_outputs, Y, X, generator)


def compute_loss(loss, outputs, target, generator):
	# for b in range(32):
	# 	print(f"{b} output: {''.join([target_tensors_to_str([o[b, :] for o in outputs])])}")
	# 	print(f"{b} target: {target_tensors_to_str(target[b, :, :])}")
	# print("-")
	if not isinstance(outputs, list):
		outputs = [outputs[:, pos, :] for pos in range(outputs.size(1))]
	cumulative_loss = 0
	idx_pad = generator.y_vocab[_PAD]
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


def compute_act_loss(outputs, act, inputs, target, generator):
	R_enc, N_enc, R_dec, N_dec = act
	idx_pad_x = generator.x_vocab["#"]
	idx_pad_y = generator.y_vocab["#"]
	idx_inputs = inputs.argmax(dim=-1)
	idx_targets = target.argmax(dim=-1)
	mask_x = (idx_inputs != idx_pad_x).type(torch.int32)
	mask_y = (idx_targets != idx_pad_y).type(torch.int32)

	rho_enc = (R_enc + N_enc) * mask_x
	rho_dec = (R_dec + N_dec) * mask_y
	return (rho_enc.mean() + rho_dec.mean()).mean()
    

def _fix_output_shape(output, Y, generator):
    # fix pred/target shape mismatch
    if output.size(1) < Y.size(1):
        missing_timesteps = Y.size(1) - output.size(1)
        pad_vecs = torch.nn.functional.one_hot(torch.tensor(generator.y_vocab['#'],
                                                            device=Y.device)).tile(output.size(0),
                                                                                   missing_timesteps, 1)
        output = torch.concat([output, pad_vecs], dim=1)
    elif output.size(1) > Y.size(1):
        output = output[:, :Y.size(1)]
    return output
	

def batch_acc(outputs, targets, vocab_size, generator):
	if isinstance(outputs, list):
		outputs = torch.concat([o.unsqueeze(1) for o in outputs], dim=1)	
	idx_pad = generator.y_vocab[_PAD]
	idx_targets = targets.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	idx_outs = outputs.argmax(dim=-1)
	out_equal_target = (idx_outs == idx_targets).type(torch.int32)
	masked_out_equal_target = out_equal_target * mask
	num_masked = (mask == 0).sum()
	num_targets = idx_targets.size(0) * idx_targets.size(1)
	return masked_out_equal_target.sum() / (num_targets - num_masked)


def batch_seq_acc(outputs, targets, generator, len_Y):
	idx_pad = generator.y_vocab[_PAD]
	idx_targets = targets.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	idx_outs = outputs.argmax(dim=-1)
	out_equal_target = (idx_outs == idx_targets).type(torch.int32)
	masked_out_equal_target = out_equal_target * mask
	num_equal_chars_per_seq = masked_out_equal_target.sum(dim=-1)
	pred_is_exact = (num_equal_chars_per_seq == torch.tensor(len_Y, device=outputs.device)).type(torch.FloatTensor)
	return pred_is_exact.mean(), pred_is_exact.std()


def get_lengths_(batch):
	EOS_idx = batch.size(2) - 2
	lengths = torch.zeros(batch.size(0)) - 1  # seq -> len
	# for each sequence, get positions where idx == EOS 
	for s, l in torch.argwhere((batch.argmax(2) == EOS_idx).type(torch.int)):
		lengths[s] = l.item() if lengths[s] == -1 else lengths[s]  # get only the first occurence for each seq (i.e., seq length)
	return lengths


def get_lengths(batch):
	def fill_missing(lengths):
		for i in range(batch.size(0)):
			if lengths[i] == -1:
				lengths[i] = batch.size(1)  # set to max len if EOS was never predicted
		return lengths

	if isinstance(batch, list):
		batch = torch.concat([o.unsqueeze(1) for o in batch], dim=1)
	lengths_batch = get_lengths_(batch)
	lengths_batch = fill_missing(lengths_batch)
	return lengths_batch


def get_len_stats(x, y):
	lengths_x = get_lengths(x)
	lengths_y = get_lengths(y)
	
	num_unequal = 0
	num_longer = 0
	num_shorter = 0
	avg_len_diff = 0
	
	num_unequal = (lengths_x != lengths_y).sum().item()
	num_longer = (lengths_x > lengths_y).sum().item()
	num_shorter = (lengths_x < lengths_y).sum().item()
	sum_len_diff = (abs(lengths_x - lengths_y)).sum().item()

	return (num_unequal, num_longer, num_shorter, sum_len_diff)


def penalty(delta):
    w = 0.5
    p = 0
    for i in range(int(delta)):
        p += w**i
    return p / 2


def batch_acc_modified(outputs, targets, vocab_size):
	targets_idx = targets.argmax(2)
	outputs = torch.concat([o.unsqueeze(1) for o in outputs], dim=1)
	idx_outs = outputs.argmax(dim=-1)
	
	lengths_x = get_lengths(outputs)
	lengths_y = get_lengths(targets)
	cut_lengths = [int(min(l_x, l_y)) for (l_x, l_y) in zip(lengths_x, lengths_y)]
	cut_targets = torch.concat([targets_idx[s, :cut_lengths[s]] for s in range(targets.size(0))])
	cut_outputs = torch.concat([idx_outs[s, :cut_lengths[s]] for s in range(targets.size(0))])
	out_equal_target = (cut_targets == cut_outputs).type(torch.float)
	deltas = torch.abs(lengths_x - lengths_y)
	penalties = deltas.type(torch.float).apply_(penalty)
	return out_equal_target.mean(), penalties.mean()