from utils.rnn_utils import make_tensor, make_padded_batch, get_token2pos
from data.data import generate_sample


def get_vocab_chars():
	return set('abcdefghijklmnopqrstuvwxyz0123456789()+*-=<>: ')


def get_vocab_size():
	return len(get_vocab_chars()) + 1


def generate_batch(length, nesting, batch_size, split='train'):
	vocab_size = get_vocab_size()
	token2pos = get_token2pos(get_vocab_chars())
	
	few_samples = [generate_sample(length=length, nesting=nesting, split=split) for i in range(batch_size)]
	
	samples_len = [len(x) for x, y in few_samples]
	targets_len = [len(y) for x, y in few_samples]

	tensor_samples = [make_tensor(x, token2pos, vocab_size) for x, y in few_samples]
	tensor_targets = [make_tensor(y, token2pos, vocab_size) for x, y in few_samples]

	padded_batch_samples = make_padded_batch(tensor_samples, samples_len, vocab_size)
	padded_batch_targets = make_padded_batch(tensor_targets, targets_len, vocab_size)

	return padded_batch_samples, padded_batch_targets, samples_len, targets_len
