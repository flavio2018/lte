from utils.rnn_utils import make_tensor, make_target_tensor, make_padded_batch
from data.data import generate_sample
import torch
from itertools import product
import pickle
import random


_EOS = '.'
_SOS = '?'
_PAD = '#'


def get_vocab_chars():
	return set('abcdefghijklmnopqrstuvwxyz0123456789()%+*-=<>[]: ')


def get_target_vocab_chars():
	return set('0123456789-') 


def get_vocab_size():
	return len(get_vocab_chars()) + 2  # carriage return, padding


def get_target_vocab_size():
	return len(get_target_vocab_chars()) + 3  # sos eos pad


def get_token2pos():
	token2pos = {t: p for p, t in enumerate(get_vocab_chars())}
	token2pos['\n'] = len(token2pos)
	token2pos[_PAD] = len(token2pos)  # padding
	return token2pos


def get_target_token2pos():
	token2pos = {t: p for p, t in enumerate(get_target_vocab_chars())}
	token2pos[_SOS] = len(token2pos)  # start of string
	token2pos[_EOS] = len(token2pos)  # end of string
	token2pos[_PAD] = len(token2pos)  # padding
	return token2pos


def get_pos2token():
	token2pos = get_token2pos()
	return {p: t for t, p in token2pos.items()}


def get_target_pos2token():
	token2pos = get_target_token2pos()
	return {p: t for t, p in token2pos.items()}


def generate_batch(max_length, max_nesting, batch_size, split='train', ops='asmif'):
	vocab_size = get_vocab_size()
	token2pos = get_token2pos()
	target_vocab_size = get_target_vocab_size()
	target_token2pos = get_target_token2pos()
	max_len_in, max_len_out = get_max_lens(max_length, max_nesting)
	
	if split != 'test':
		few_samples = [generate_sample(length=torch.randint(1, max_length+1, (1,)).item(),
									   nesting=torch.randint(1, max_nesting+1, (1,)).item(),
									   split=split, ops=ops) for i in range(batch_size)]
	else:
		few_samples = [generate_sample(length=max_length,
									   nesting=max_nesting,
									   split=split, ops=ops) for i in range(batch_size)]
	# revert target
	# few_samples = [(x, y[::-1]) for x, y in few_samples]

	samples_len = [len(x) for x, y in few_samples]
	targets_len = [len(y) + 2 for x, y in few_samples]  # targets start with SOS and end with EOS

	tensor_samples = [make_tensor(x, token2pos, vocab_size) for x, y in few_samples]
	tensor_targets = [make_target_tensor(y, target_token2pos, target_vocab_size) for x, y in few_samples]

	padded_batch_samples = make_padded_batch(tensor_samples, samples_len, vocab_size, max(samples_len))
	padded_batch_targets = make_padded_batch(tensor_targets, targets_len, target_vocab_size, max(targets_len))

	return padded_batch_samples, padded_batch_targets, samples_len, targets_len


def get_max_lens(max_length, max_nesting):
	# return max_length * 2 + 3, max_length + 2
	outer_nests_term = (max_length * 3 + 10) * (max_nesting - 1)
	inner_nest_term = max_length * 4 + 10
	nest_tok_term = (max_nesting - 1) * 2
	_num_steps_out = max_length * 2 + 1  # for padding
	_num_steps = outer_nests_term + inner_nest_term + nest_tok_term
	# num_steps = max(_num_steps, _num_steps_out)    
	return _num_steps, _num_steps_out


class SubsetDataset:
	def __init__(self, max_len, max_nes, ops):
		self.max_len = max_len
		self.max_nes = max_nes
		self.ops = ops
		self.subsets = {}
		self._load_subsets()

	def _load_subsets(self):
		print("Loading subsets...")
		for l, n in product(range(1, self.max_len+1), range(1, self.max_nes+1)):
			for split in ['train', 'valid', 'test']:
			    with open(f'/home/fpetruzzellis/Progetti/lte/data/subsets/{self.ops}_len{l}_nes{n}_{split}.pickle', 'rb') as f:
			        self.subsets[f"len{l}nes{n}_{split}"] = list(pickle.load(f))
		print("Subsets loaded.")

	def generate_batch(self, max_length, max_nesting, batch_size, split='train', ops='asmif', mod=3):
		vocab_size = get_vocab_size()
		token2pos = get_token2pos()
		target_vocab_size = get_target_vocab_size()
		target_token2pos = get_target_token2pos()
		
		if split != 'test':
			few_samples = [self.generate_sample(length=torch.randint(1, max_length+1, (1,)).item(),
			 						   			nesting=torch.randint(1, max_nesting+1, (1,)).item(),
			 						   			split=split, ops=ops, mod=mod) for i in range(batch_size)]
		else:
			few_samples = [self.generate_sample(length=max_length,
												nesting=max_nesting,
												split=split, ops=ops, mod=mod) for i in range(batch_size)]

		samples_len = [len(x) for x, y in few_samples]
		targets_len = [len(y) + 2 for x, y in few_samples]  # targets start with SOS and end with EOS

		tensor_samples = [make_tensor(x, token2pos, vocab_size) for x, y in few_samples]
		tensor_targets = [make_target_tensor(y, target_token2pos, target_vocab_size) for x, y in few_samples]

		padded_batch_samples = make_padded_batch(tensor_samples, samples_len, vocab_size, max(samples_len))
		padded_batch_targets = make_padded_batch(tensor_targets, targets_len, target_vocab_size, max(targets_len))

		return padded_batch_samples, padded_batch_targets, samples_len, targets_len

	def generate_sample(self, length, nesting, split='train', ops='a', mod=3):
		return random.choice(self.subsets[f"len{length}nes{nesting}_{split}"])
