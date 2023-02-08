from collections import OrderedDict
import string
import re
import torch
import torch.nn.functional as F
from torchtext.vocab import vocab
from torchtext.transforms import ToTensor, VocabTransform
from data.data import generate_sample


_EOS = '.'
_SOS = '?'
_PAD = '#'


class LTEGenerator:
    def __init__(self, device):
        x_vocab_chars = string.ascii_lowercase + string.digits + '()%+*-=<>[]: '
        y_vocab_chars = string.digits + '-'
        self.x_vocab = vocab(
            OrderedDict([(c, 1) for c in x_vocab_chars]),
            specials=[_PAD],
            special_first=False)
        self.y_vocab = vocab(
            OrderedDict([(c, 1) for c in y_vocab_chars]),
            specials=[_SOS, _EOS, _PAD],
            special_first=False)
        self.x_vocab_trans = VocabTransform(self.x_vocab)
        self.y_vocab_trans = VocabTransform(self.y_vocab)
        self.x_to_tensor_trans = ToTensor(padding_value=self.x_vocab[_PAD])
        self.y_to_tensor_trans = ToTensor(padding_value=self.y_vocab[_PAD])
        self.device = torch.device(device)

    def _generate_sample(self, max_length, max_nesting, split, ops, batch_size):
        return generate_sample(length=torch.randint(1, max_length+1, (1,)).item(),
                               nesting=torch.randint(1, max_nesting+1, (1,)).item(),
                               split=split,
                               ops=ops)

    def _generate_sample_naive(self, length, nesting, split, ops, batch_size):
        return generate_sample(length=length,
                               nesting=nesting,
                               split=split,
                               ops=ops)
    
    def generate_batch(self, max_length, max_nesting, batch_size, split='train', ops='asmif'):
        samples, targets = [], []
        samples_len, targets_len = [], []

        for _ in range(batch_size):
            if split == 'test':
                x, y = self._generate_sample_naive(max_length, max_nesting, split, ops, batch_size)
            else:
                x, y = self._generate_sample(max_length, max_nesting, split, ops, batch_size)
            samples.append(list(x))
            targets.append([_SOS] + list(y) + [_EOS])
            samples_len.append(len(samples[-1]))
            targets_len.append(len(targets[-1]))
        
        tokenized_samples = self.x_vocab_trans(samples)
        tokenized_targets = self.y_vocab_trans(targets)
        padded_samples = self.x_to_tensor_trans(tokenized_samples).to(self.device)
        padded_targets = self.y_to_tensor_trans(tokenized_targets).to(self.device)
        return (F.one_hot(padded_samples).type(torch.float),
                F.one_hot(padded_targets).type(torch.float),
                samples_len,
                targets_len)
    
    def x_to_str(self, batch):
        return [''.join(self.x_vocab.lookup_tokens(tokens)) for tokens in batch.argmax(-1).tolist()]
    
    def y_to_str(self, batch):
        return [''.join(self.y_vocab.lookup_tokens(tokens)) for tokens in batch.argmax(-1).tolist()]


class LTEStepsGenerator(LTEGenerator):
    def __init__(self, device):
        super().__init__(device)
        vocab_chars = string.ascii_lowercase + string.digits + '()%+*-=<>[]: '
        self.x_vocab = vocab(
            OrderedDict([(c, 1) for c in vocab_chars]),
            specials=[_PAD],
            special_first=False)
        self.y_vocab = vocab(
            OrderedDict([(c, 1) for c in vocab_chars]),
            specials=[_SOS, _EOS, _PAD],
            special_first=False)
        self.x_vocab_trans = VocabTransform(self.x_vocab)
        self.y_vocab_trans = VocabTransform(self.y_vocab)
        self.x_to_tensor_trans = ToTensor(padding_value=self.x_vocab[_PAD])
        self.y_to_tensor_trans = ToTensor(padding_value=self.y_vocab[_PAD])
        self.device = torch.device(device)

    def _generate_sample_naive(self, length, nesting, split, ops, batch_size):
        return generate_sample(length=length,
                               nesting=nesting,
                               split=split,
                               ops=ops,
                               steps=True)

    def _generate_sample(self, max_length, max_nesting, split, ops, batch_size):
        return generate_sample(length=torch.randint(1, max_length+1, (1,)).item(),
                               nesting=torch.randint(1, max_nesting+1, (1,)).item(),
                               split=split,
                               ops=ops,
                               steps=True)

    def generate_batch(self, max_length, max_nesting, batch_size, split='train', ops='asmif', start_to_end=False, simplify=False):

        samples, targets = [], []
        samples_len, targets_len = [], []
        subexpr_start_end = []

        for _ in range(batch_size):
            if split == 'test':
                _, _, steps, values = self._generate_sample_naive(max_length, max_nesting, split, ops, batch_size)
                if start_to_end:
                    x, y = steps[0], values[-1]
                elif simplify:
                    start_end = [self._get_start_end_expr(e) for e in steps[:-1]]
                    subexpressions = [step[s:e] for (s,e), step in zip(start_end, steps[:-1])]
                    x, y = steps[0], subexpressions[0]
                else:
                    x, y = steps[0], values[0]
            else:
                _, _, steps, values = self._generate_sample(max_length, max_nesting, split, ops, batch_size)
                rand_idx = torch.randint(0, len(steps)-1, (1,)).item()
                if start_to_end:
                    x, y = steps[0], values[-1]
                elif simplify:
                    start_end = [self._get_start_end_expr(e) for e in steps[:-1]]
                    subexpressions = [step[s:e] for (s,e), step in zip(start_end, steps[:-1])]
                    x, y = steps[rand_idx], subexpressions[rand_idx]
                else:
                    x, y = steps[rand_idx], values[rand_idx]
            subexpr_start_end += [self._get_start_end_expr(x)]
            samples.append(list(x))
            targets.append([_SOS] + list(y) + [_EOS])
            samples_len.append(len(samples[-1]))
            targets_len.append(len(targets[-1]))
        
        batch_x = self._build_batch(samples)
        batch_y = self._build_batch(targets, y=True)
        return (batch_x,
                batch_y,
                samples_len,
                targets_len,
                self._get_subexp_mask(subexpr_start_end, samples))

    def _build_batch(self, samples, y=False):
        if not y:
            tokenized_samples = self.x_vocab_trans(samples)
            padded_samples = self.x_to_tensor_trans(tokenized_samples).to(self.device)
            return F.one_hot(padded_samples, num_classes=len(self.x_vocab)).type(torch.float)
        else:
            tokenized_targets = self.y_vocab_trans(samples)
            padded_targets = self.y_to_tensor_trans(tokenized_targets).to(self.device)
            return F.one_hot(padded_targets, num_classes=len(self.y_vocab)).type(torch.float)

    def _get_start_end_expr(self, expression):
        match = re.search(r'[(][a-z0-9+*\-:=<>\[\] ]+[)]', expression)
        return match.start(0), match.end(0)

    def _get_subexp_mask(self, x_start_end, samples):
        tokenized_samples = self.x_vocab_trans(samples)
        batch = self.x_to_tensor_trans(tokenized_samples).to(self.device)
        mask = torch.ones((batch.size(0), batch.size(1)))
        for row, (start, end) in enumerate(x_start_end):
            mask[row, start:end] -= 1
        return mask.type(torch.BoolTensor).to(self.device)
