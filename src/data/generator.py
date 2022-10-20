from utils.rnn_utils import make_tensor, make_padded_batch
from data.data import generate_sample


def get_vocab_chars():
	return set('abcdefghijklmnopqrstuvwxyz0123456789()+*-=<>: ')

def get_vocab_size():
	return len(get_vocab_chars()) + 1

def get_token2pos():
	vocab_chars = get_vocab_chars()
	token2pos = {t: p for p, t in enumerate(vocab_chars)}
	token2pos['\n'] = len(token2pos)
	return token2pos

def get_pos2token():
	token2pos = get_token2pos()
	return {p: t for t, p in token2pos.items()}

def generate_batch(length, nesting, batch_size, split='train'):
	vocab_size = get_vocab_size()
	token2pos = get_token2pos()
	
	few_samples = [generate_sample(length=length, nesting=nesting, split=split) for i in range(batch_size)]
	
	samples_len = [len(x) for x, y in few_samples]
	targets_len = [len(y) for x, y in few_samples]

	tensor_samples = [make_tensor(x, token2pos, vocab_size) for x, y in few_samples]
	tensor_targets = [make_tensor(y, token2pos, vocab_size) for x, y in few_samples]

	padded_batch_samples = make_padded_batch(tensor_samples, samples_len, vocab_size)
	padded_batch_targets = make_padded_batch(tensor_targets, targets_len, vocab_size)

	return padded_batch_samples, padded_batch_targets, samples_len, targets_len
