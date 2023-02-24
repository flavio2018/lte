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


@hydra.main(config_path="../../conf/local", config_name="test_start2end", version_base='1.2')
def main(cfg):
	warnings.filterwarnings('always', category=UserWarning)
	lte, lte_kwargs = build_generator(cfg)
	model = load_model(cfg, lte)
	wrapped_model = ModelWrapper(model)

	ax = test_ood_start2end(wrapped_model, lte, 10, generator_kwargs={'batch_size': cfg.bs,
																	 'start_to_end': cfg.start_to_end,
																	 'split': 'test',
																	 'ops': cfg.ops})
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.ckpt[:-4]}_start2end.pdf"))	

def contain_space(outputs):
	return np.char.count(outputs, ' ') == 1

def cut_at_first_dot(repl):
	cut_idx = re.search(r'\.', repl).span(0)[0]  # postion of first dot
	return repl[:cut_idx]

def inputs_contain_substrings(inputs, outputs, running):
	inputs_contain_substrings = []
	
	for idx, r in enumerate(running):
		if r:
			_, substring = outputs[idx].split()
			inputs_contain_substrings += [cut_at_first_dot(substring) in inputs[idx]]
		else:
			inputs_contain_substrings += [r]

	return np.array(inputs_contain_substrings)

def replace_substrings_in_inputs(inputs, outputs, running):
	next_inputs = []

	for idx, r in enumerate(running):
		if r:
			result, substring = outputs[idx].split()
			next_inputs.append(inputs[idx].replace(cut_at_first_dot(substring), result))
		else:
			next_inputs.append('#')
		   
	return next_inputs

def model_output_to_next_input(cur_input, output, running):
	chararray_outputs = np.array(output)
	chararray_inputs = np.array([x.replace('#', '') for x in cur_input])
	
	# check output structure
	outputs_are_well_formed = contain_space(chararray_outputs)
	running &= outputs_are_well_formed
	
	# check substring in input
	inputs_do_contain_substrings = inputs_contain_substrings(chararray_inputs, chararray_outputs, running)
	running &= inputs_do_contain_substrings

	# substitute
	next_input = replace_substrings_in_inputs(chararray_inputs,
											  chararray_outputs,
											  running)
	return next_input, running

class ModelWrapper:
	
	def __init__(self, model):
		self.model = model
		self.running = []

	def __call__(self, X, Y=None, tf=False, max_nes=0):
		self.model.eval()
		self.running = []
		lte = self.model.generator
		running = np.array([True]*X.size(0))
		original_batch = X
		
		for cur_nes in range(max_nes):
			# Y = Y if (cur_nes == (max_nes - 1)) else None
			output = self.model(X, Y=None, tf=tf)
			next_inputs, running = model_output_to_next_input(lte.x_to_str(X),
															  lte.y_to_str(output),
															  running)
			X = lte._build_batch([list(i) for i in next_inputs])
			self.running.append(running)
		return lte._build_batch([list(i) + ['.'] for i in next_inputs], y=True)

def test_ood_start2end(model, generator, max_nes, tf=False, generator_kwargs=None, plot_ax=None, plot_label=None):
	accuracy_values = []
	nesting_values = []
	survivors = []
	
	for n in range(1, max_nes+1):
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
				warnings.warn(f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing.")
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

	ax = sns.barplot(data=df, x='Nesting', y='Character Accuracy', label=plot_label, ax=plot_ax)
	for i, s in enumerate(survivors):
		ax.annotate("{:.2f}%".format(s/generator_kwargs['batch_size'] * 100), xy=(i, df.iloc[i, 0]), ha='center')
	return ax

if __name__ == "__main__":
	main()
