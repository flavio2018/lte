import warnings
import pandas as pd
from model.test import batch_acc, batch_seq_acc, _fix_output_shape
from data.generator import LTEStepsGenerator


def main():
	parse_outputs()
	df_nes_1to6 = pd.read_csv('../../chatgpt/ans/2023-March-14_15-50-17_text-davinci-003.csv')
	df_nes_7to10 = pd.read_csv('../../chatgpt/ans/2023-March-15_09-40-29_text-davinci-003.csv')
	df = pd.concat([df_nes_1to6, df_nes_7to10])
	lte = LTEStepsGenerator(device='cuda')

	acc_values = []
	acc_std = []
	seq_acc_values = []
	seq_acc_std = []

	for n in range(1, 11):
		targets = df[df['Nesting'] == n]['Target'].astype(str).values
		targets_batch = lte._build_batch([list(t) for t in targets], y=True)
		outputs = df[df['Nesting'] == n]['Output'].values
		outputs_batch = lte._build_batch([list(o) for o in outputs], y=True)

		if outputs_batch.size() != targets_batch.size():
			warn_str = f"Outputs shape {outputs_batch.size()} different from targets shape {targets_batch.size()}. Fixing."
			warnings.warn(warn_str)
			print(targets)
			print(outputs)
			outputs_batch = _fix_output_shape(outputs_batch, targets_batch, lte)

		for slice_idx in range(0, 100, 10):
			avg_acc, std_acc = batch_acc(outputs_batch[slice_idx:slice_idx+100], targets_batch[slice_idx:slice_idx+100], targets_batch.size(-1), lte)


def parse_outputs():
	filenames = ['2023-March-14_15-50-17_text-davinci-003.txt',
				'2023-March-15_09-40-29_text-davinci-003.txt']

	for filename in filenames:
		inputs = { n: [] for n in range(1, 11) }
		targets = { n: [] for n in range(1, 11) }
		outputs = { n: [] for n in range(1, 11) }
	
		with open(f'../../chatgpt/ans/{filename}', 'r') as gpt_out_f:
			for line_idx, line in enumerate(gpt_out_f):
				if line_idx % 3 == 0:
					# example prompt
					pass
				elif line_idx % 3 == 1:
					expression = line.split('=')[0].strip()
					nesting = expression.count('(')
					inputs[nesting].append(expression)
					targets[nesting].append(eval(expression))
				else:
					outputs[nesting].append(line.strip())

		full_inputs = { n: v for n, v in inputs.items() if len(v) >= 100 }
		full_targets = { n: v for n, v in targets.items() if len(v) >= 100 }
		full_outputs = { n: v for n, v in outputs.items() if len(v) >= 100 }

		df_data = [[i, t, o, n]
			for n in full_inputs
			for i, t, o in zip(full_inputs[n][:100], full_targets[n][:100], full_outputs[n][:100])]
		df = pd.DataFrame(df_data, columns=['Input', 'Target', 'Output', 'Nesting'])
		df.to_csv(f"../../chatgpt/ans/{filename.split('.')[0]}.csv")


if __name__ == '__main__':
	main()
