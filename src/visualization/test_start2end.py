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

	ax = test_ood_start2end(wrapped_model, lte, 10, generator_kwargs={'batch_size': cfg.bs,
																	 'start_to_end': cfg.start_to_end,
																	 'filtered_s2e': cfg.filtered_s2e,
																	 'split': 'test',
																	 'ops': cfg.ops})
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.ckpt[:-4]}_start2end.pdf"))	

def contain_one_space(outputs):
	return np.char.count(outputs, ' ') == 1

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

def model_output_to_next_input(cur_input, output, output_tensor, running, use_tricks):
	chararray_outputs = np.array(output)
	chararray_inputs = np.array([x.replace('#', '') for x in cur_input])

	# check output structure
	outputs_have_stopped = have_stopped(chararray_outputs)
	logging.info(f"{(~outputs_have_stopped).sum()} outputs have not stopped.")
	running &= outputs_have_stopped
	logging.info(f"{running.sum()} outputs are running.")
	chararray_outputs = cut_at_first_dot(chararray_outputs, running)
	outputs_are_well_formed = contain_one_space(chararray_outputs)
	logging.info(f"{(~outputs_are_well_formed & running).sum()} outputs are not well formed.")
	if use_tricks:
		chararray_outputs = replace_double_spaces(chararray_outputs)
		outputs_are_well_formed = contain_one_space(chararray_outputs)
		logging.info(f"{(~outputs_are_well_formed & running).sum()} outputs are not well formed after double space correction.")
	logging.info('\n'.join([f"{i} → {o}" for i, o in zip(chararray_inputs[~outputs_are_well_formed & running][:20], 
		chararray_outputs[~outputs_are_well_formed & running][:20])]))
	logging.info("Top 2 logits for first 10 ill-formed model outputs")
	top2_logits, _ = output_tensor[torch.tensor(~outputs_are_well_formed & running,
											 device=output_tensor.device)][:10].topk(k=2, dim=-1)
	logging.info(top2_logits)
	running &= outputs_are_well_formed
	logging.info(f"{running.sum()} outputs are running.")
	
	# check substring in input
	inputs_do_contain_substrings = inputs_contain_substrings(chararray_inputs, chararray_outputs, running)
	logging.info(f"{(~inputs_do_contain_substrings & running).sum()} outputs have wrong substrings.")
	logging.info('\n'.join([f"{i} → {o}" for i, o in zip(chararray_inputs[~inputs_do_contain_substrings & running][:20], 
		chararray_outputs[~inputs_do_contain_substrings & running][:20])]))
	logging.info("Top 2 logits for first 10 no-substring model outputs")
	top2_logits, _ = output_tensor[torch.tensor(~inputs_do_contain_substrings & running,
											 device=output_tensor.device)][:10].topk(k=2, dim=-1)
	logging.info(top2_logits)
	if use_tricks:
		inputs_do_soft_contain_substrings = inputs_soft_contain_substrings(chararray_inputs, chararray_outputs, running)
		logging.info(f"{(~inputs_do_soft_contain_substrings & running).sum()} outputs have non-softmatching substrings.")
		running &= inputs_do_soft_contain_substrings
		next_input = soft_replace_substrings_in_inputs(chararray_inputs,
													   chararray_outputs,
													   running)
	else:
		running &= inputs_do_contain_substrings
		next_input = replace_substrings_in_inputs(chararray_inputs,
												  chararray_outputs,
												  running)
	logging.info(f"{running.sum()} outputs are running.")
	return next_input, running

class ModelWrapper:
	
	def __init__(self, model):
		self.model = model
		self.running = []
		self.use_tricks = False

	def __call__(self, X, Y=None, tf=False, max_nes=0):
		self.model.eval()
		self.running = []
		lte = self.model.generator
		running = np.array([True]*X.size(0))
		original_batch = X
		
		for cur_nes in range(max_nes):
			logging.info(f"~~~ cur_nes {cur_nes} ~~~")
			# Y = Y if (cur_nes == (max_nes - 1)) else None
			output = self.model(X, Y=None, tf=tf)
			next_inputs, running = model_output_to_next_input(lte.x_to_str(X),
															  lte.y_to_str(output),
															  output,
															  running,
															  self.use_tricks)
			X = lte._build_batch([list(i) for i in next_inputs])
			self.running.append(running)
		return lte._build_batch([list(i) + ['.'] for i in next_inputs], y=True)

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
	ax = sns.lineplot(x=range(1, max_nes-1), y=[s/generator_kwargs['batch_size'] for s in survivors], marker='o', color='tab:cyan')
	return ax

if __name__ == "__main__":
	main()
