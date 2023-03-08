import hydra
import omegaconf
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime as dt
import os
from data.generator import LTEGenerator, LTEStepsGenerator, get_mins_maxs_from_mask
from model.regression_tran import UTwRegressionHead
from model.ut.UniversalTransformer import UniversalTransformer
from model.test import batch_acc, _fix_output_shape
from output_dfa import output_dfa
from visualization.test_ood import build_generator, load_model
import warnings
import logging


@hydra.main(config_path="../../conf/local", config_name="test_start2end", version_base='1.2')
def main(cfg):
	warnings.filterwarnings('always', category=UserWarning)
	now_day = dt.now().strftime('%Y-%m-%d')
	now_time = dt.now().strftime('%H:%M')
	logging.basicConfig(filename=os.path.join(hydra.utils.get_original_cwd(), f'../logs/{now_day}_{now_time}_{cfg.ckpt[:-4]}_test_start2end.txt'),
			filemode='a',
			format='%(message)s',
			datefmt='%H:%M:%S',
			level=logging.INFO)
	lte, lte_kwargs = build_generator(cfg)
	model = load_model(cfg, lte)
	wrapped_model = ModelWrapper(model)
	wrapped_model.use_tricks = cfg.tricks
	tricks = '_tricks' if cfg.tricks else ''

	ax, df = test_ood_start2end(wrapped_model, lte, 10, generator_kwargs={'batch_size': cfg.bs,
																	 'start_to_end': cfg.start_to_end,
																	 'filtered_s2e': cfg.filtered_s2e,
																	 'split': 'test',
																	 'ops': cfg.ops})
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.ckpt[:-4]}_start2end{tricks}.pdf"))
	df["Character Accuracy"].T.to_latex(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/tables/{cfg.ckpt[:-4]}_start2end{tricks}.tex"))	

def contain_one_space(outputs):
	return (np.char.count(outputs, ' ', start=1, end=-1) == 1)

def replace_double_spaces(outputs):
	return np.char.replace(outputs, '  ', ' ')

def cut_at_first_dot(outputs, running):
	cut_outputs = []

	for idx, r in enumerate(running):
		if r:
			cut_idx = re.search(r'\.', outputs[idx]).span(0)[0]  # postion of first dot
			cut_outputs.append(outputs[idx][:cut_idx])
		else:
			cut_outputs.append(outputs[idx])
	return np.array(cut_outputs)

def have_stopped(outputs):
	return np.array(['.' in o for o in outputs])

def inputs_contain_substrings(inputs, outputs, running):
	inputs_contain_substrings = []
	
	for idx, r in enumerate(running):
		if r:
			_, substring = outputs[idx].split()
			inputs_contain_substrings += [substring in inputs[idx]]
		else:
			inputs_contain_substrings += [r]

	return np.array(inputs_contain_substrings)

def inputs_soft_contain_substrings(inputs, outputs, running):
	inputs_soft_contain_substrings = []
	substring_re = re.compile(r'[(][a-z0-9+*\-:=<>\[\] ]+[)]')

	for idx, r in enumerate(running):
		if r:
			_, candidate = outputs[idx].split()
			if candidate in inputs[idx]:
				inputs_soft_contain_substrings += [True]
			else:
				input_substring = substring_re.findall(inputs[idx])[0]
				if levenshteinDistance(input_substring, candidate) <= 2:
					inputs_soft_contain_substrings += [True]
				else:
					inputs_soft_contain_substrings += [False]
		else:
			inputs_soft_contain_substrings += [r]
	return np.array(inputs_soft_contain_substrings)

def replace_substrings_in_inputs(inputs, outputs, running):
	next_inputs = []

	for idx, r in enumerate(running):
		if r:
			result, substring = outputs[idx].split()
			next_inputs.append(inputs[idx].replace(substring, result))
		else:
			next_inputs.append('()')
		   
	return next_inputs

def levenshteinDistance(s1, s2):
	if len(s1) > len(s2):
		s1, s2 = s2, s1

	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2+1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

def soft_replace_substrings_in_inputs(inputs, outputs, running):
	next_inputs = []
	substring_re = re.compile(r'[(][a-z0-9+*\-:=<>\[\] ]+[)]')

	for idx, r in enumerate(running):
		if r:
			result, output_substring = outputs[idx].split()
			input_substring = substring_re.findall(inputs[idx])[0]
			next_inputs.append(inputs[idx].replace(input_substring, result))
		else:
			next_inputs.append('()')
	return next_inputs

class ModelWrapper:
	
	def __init__(self, model):
		self.model = model
		self.running = []
		self.use_tricks = False
		self.use_dfa = False

	def __call__(self, X, Y=None, tf=False, max_nes=0):
		self.model.eval()
		self.running = []
		lte = self.model.generator
		running = np.array([True]*X.size(0))
		
		for cur_nes in range(max_nes):
			logging.info(f"\n~~~ cur_nes {cur_nes} ~~~")
			# Y = Y if (cur_nes == (max_nes - 1)) else None
			if self.use_dfa:
				output = self.fwd_dfa(X, tf=tf)
			else:
				output = self.model(X, Y=None, tf=tf)
			next_inputs, running = self.model_output_to_next_input(X, output, running)
			X = lte._build_batch([list(i) for i in next_inputs])
			self.running.append(running)
		return lte._build_batch([list(i) + ['.'] for i in next_inputs], y=True)

	def fwd_dfa(self, X, tf=False):
		it, max_it = 0, 100
		lte = self.model.generator
		
		encoding, src_mask = self.model._test_fwd_encoder_step(X)
		stopped = torch.zeros(X.size(0)).type(torch.BoolTensor).to(X.device)
		Y_pred_v = torch.tile(F.one_hot(torch.tensor([lte.y_vocab['?']]), num_classes=len(lte.y_vocab)), dims=(X.size(0), 1, 1)).type(torch.FloatTensor).to(X.device)
		output = Y_pred_v
		
		while not stopped.all() and (it < max_it):
			it += 1
			Y_pred = self.model._test_fwd_decode_step(encoding, src_mask, Y_pred_v)
			output = torch.concat([output, Y_pred], dim=1)     

			# logits to tokens conversion 
			pred_idx = Y_pred.argmax(-1) 
			Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)

			# equivalent to no-dfa
			pred_idx = Y_pred.argmax(-1) 
			Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
			stopped = torch.logical_or((pred_idx.squeeze() == EOS_idx), stopped)
		return output[:, 1:, :]

	def model_output_to_next_input(self, X, output_tensor, running):
		lte = self.model.generator
		y_vocab_itos = lte.y_vocab.get_itos()
		itos_f = np.vectorize(lambda x: y_vocab_itos[x])
		cur_input, output = lte.x_to_str(X), lte.y_to_str(output_tensor)
		chararray_outputs = np.array(output)
		chararray_inputs = np.array([x.replace('#', '') for x in cur_input])

		# check output structure
		outputs_have_stopped = have_stopped(chararray_outputs)
		running &= outputs_have_stopped
		chararray_outputs = cut_at_first_dot(chararray_outputs, running)
		max_cut_length = max([len(o) for o in chararray_outputs])
		
		logging.info(f"{(~outputs_have_stopped).sum()} outputs have not stopped.")
		logging.info(f"{running.sum()} outputs are running.")
		
		if self.use_tricks:
			chararray_outputs = replace_double_spaces(chararray_outputs)
			outputs_are_well_formed = contain_one_space(chararray_outputs)
			logging.info(f"\n{(~outputs_are_well_formed & running).sum()} outputs are not well formed after double space correction.")
		else:
			outputs_are_well_formed = contain_one_space(chararray_outputs)
			logging.info(f"\n{(~outputs_are_well_formed & running).sum()} outputs are not well formed.")

		if (~outputs_are_well_formed & running).sum() > 0:
			notwell_formed_running_inputs = chararray_inputs[~outputs_are_well_formed & running]
			num_log_idx = 10 if notwell_formed_running_inputs.shape[0] > 10 else notwell_formed_running_inputs.shape[0]
			log_idx = np.random.choice(notwell_formed_running_inputs.shape[0], size=num_log_idx, replace=False)
			top2_logits, top2_idx = output_tensor[torch.tensor(~outputs_are_well_formed & running, device=output_tensor.device)][torch.tensor(log_idx[:10])].topk(k=2, dim=-1)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(notwell_formed_running_inputs[log_idx], chararray_outputs[~outputs_are_well_formed & running][log_idx])]))
			logging.info("\nTop 2 logits & predictions for first 10 ill-formed model outputs")
			logging.info('\n\n'.join([f"{logits[:max_cut_length]}\n{idx[:max_cut_length]}"
				for logits, idx in zip(top2_logits.cpu().numpy().round(decimals=2), itos_f(top2_idx.cpu().numpy()))]))
			
		running &= outputs_are_well_formed
		logging.info(f"{running.sum()} outputs are running.")
		
		# check substring in input
		inputs_do_contain_substrings = inputs_contain_substrings(chararray_inputs, chararray_outputs, running)
		logging.info(f"\n{(~inputs_do_contain_substrings & running).sum()} outputs have wrong substrings.")
		
		if (~inputs_do_contain_substrings & running).sum() > 0:
			inputs_without_substring_running = chararray_inputs[~inputs_do_contain_substrings & running]
			num_log_idx = 10 if inputs_without_substring_running.shape[0] > 10 else inputs_without_substring_running.shape[0]
			log_idx = np.random.choice(inputs_without_substring_running.shape[0], size=num_log_idx, replace=False)
			top2_logits, top2_idx = output_tensor[torch.tensor(~inputs_do_contain_substrings & running, device=output_tensor.device)][torch.tensor(log_idx[:10])].topk(k=2, dim=-1)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(inputs_without_substring_running[log_idx], chararray_outputs[~inputs_do_contain_substrings & running][log_idx])]))
			logging.info("\nTop 2 logits & predictions for first 10 no-substring model outputs")
			logging.info('\n\n'.join([f"{logits[:max_cut_length]}\n{idx[:max_cut_length]}"
				for logits, idx in zip(top2_logits.cpu().numpy().round(decimals=2), itos_f(top2_idx.cpu().numpy()))]))
		
		if self.use_tricks:
			inputs_do_soft_contain_substrings = inputs_soft_contain_substrings(chararray_inputs, chararray_outputs, running)
			logging.info(f"\n{(~inputs_do_soft_contain_substrings & running).sum()} outputs have non-softmatching substrings.")
			running &= inputs_do_soft_contain_substrings
			next_input = soft_replace_substrings_in_inputs(chararray_inputs, chararray_outputs, running)
		else:
			running &= inputs_do_contain_substrings
			next_input = replace_substrings_in_inputs(chararray_inputs, chararray_outputs, running)

		logging.info(f"\n{running.sum()} outputs are running.")
		
		return next_input, running

def test_ood_start2end(model, generator, max_nes, tf=False, generator_kwargs=None, plot_ax=None, plot_label=None):
	accuracy_values = []
	nesting_values = []
	survivors = []

	for n in range(3, max_nes+1):
		logging.info(f"\n--- nesting {n} ---")
		values = generator.generate_batch(1, n, **generator_kwargs)

		if isinstance(generator, LTEStepsGenerator):
			X, Y, lenX, lenY, mask = values
		else:
			X, Y, lenX, lenY = values

		with torch.no_grad():
			output = model(X, Y, tf=tf, max_nes=n)
		
		running = model.running[-1]
		if running.any():
			output, Y = output[running], Y[running]
			if output.size() != Y[:, 1:].size():
				warn_str = f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing."
				logging.info(warn_str)
				warnings.warn(warn_str)
				output = _fix_output_shape(output, Y[:, 1:], generator)

			acc = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
			accuracy_values += [acc.item()]
		else:
			accuracy_values += [0]
		nesting_values += [n]
		survivors += [running.sum()]

	df = pd.DataFrame()
	df['Character Accuracy'] = accuracy_values
	df['Nesting'] = nesting_values

	ax = sns.barplot(data=df, x='Nesting', y='Character Accuracy', label=plot_label, ax=plot_ax, color='tab:blue')
	ax = sns.lineplot(x=range(max_nes-2), y=[s/generator_kwargs['batch_size'] for s in survivors], marker='o', color='tab:cyan')
	return ax, df

if __name__ == "__main__":
	main()
