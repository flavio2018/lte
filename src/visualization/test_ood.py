import hydra
import omegaconf
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from data.generator import LTEGenerator, LTEStepsGenerator, get_mins_maxs_from_mask
from model.regression_tran import UTwRegressionHead
from model.ut.UniversalTransformer import UniversalTransformer
from model.test import batch_acc, batch_seq_acc, _fix_output_shape
import warnings
import logging
import wandb


@hydra.main(config_path="../../conf/local", config_name="test_ood", version_base='1.2')
def main(cfg):
	model_id = 'regr_ut' if cfg.regr_ut else 'ut'
	task_id = 'simplify_w_value' if cfg.simplify_w_value else 'simplify'
	warnings.filterwarnings("always", category=UserWarning)
	now_day = dt.now().strftime('%Y-%m-%d')
	now_time = dt.now().strftime('%H:%M')
	logging.basicConfig(filename=os.path.join(hydra.utils.get_original_cwd(), f'../logs/{now_day}_{now_time}_{cfg.ckpt[:-4]}_test_ood.txt'),
			filemode='a',
			format='%(message)s',
			datefmt='%H:%M:%S',
			level=logging.INFO)

	wandb.init(
		project="neurips23",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"))
	wandb.run.name = 'test_ood'
	lte, lte_kwargs = build_generator(cfg)
	model = load_model(cfg, lte)

	if cfg.tables:
		metric = 'characc'
		ax, df = test_ood(model, lte, 'Nesting', use_y=cfg.use_y, tf=cfg.tf, generator_kwargs=lte_kwargs)
		plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
			f"../reports/figures/{cfg.ckpt[:-4]}_{task_id}_{model_id}_{metric}.pdf"))
		df = df.set_index('Nesting')
		df = np.round(df, 5)
		df.T.to_latex(os.path.join(hydra.utils.get_original_cwd(),
			f"../reports/tables/{cfg.ckpt[:-4]}_{task_id}_{model_id}_{metric}.tex"))	
		if isinstance(model, UTwRegressionHead):
			plt.clf()
			metric = 'huberloss'
			ax, _ = test_ood(model, lte, 'nesting', use_y=cfg.use_y, tf=cfg.tf, generator_kwargs=lte_kwargs, regr=True)
			plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
				f"../reports/figures/{cfg.ckpt[:-4]}_{task_id}_{model_id}_{metric}.pdf"))

	if cfg.plot_attn:
		plot_attn(model, lte, generator_kwargs=lte_kwargs)


def build_generator(cfg):
	if cfg.step_generator:
		lte = LTEStepsGenerator(cfg.device, cfg.same_vocab)
		lte_kwargs = {
			"batch_size": cfg.bs,
			"simplify_w_value": cfg.simplify_w_value,
			"filtered_swv": cfg.filtered_swv,
			"start_to_end": cfg.start_to_end,
			"filtered_s2e": cfg.filtered_s2e,
			"split": "test",
			"ops": cfg.ops,
		}
	else:
		lte = LTEGenerator(cfg.device)
		lte_kwargs = {
			"batch_size": cfg.bs,
			"split": "test",
			"ops": cfg.ops,
		}
	return lte, lte_kwargs


def load_model(cfg, lte):
	if cfg.regr_ut:
		 model = UTwRegressionHead(d_model=cfg.d_model,
								 num_heads=cfg.num_heads,
								 num_layers=cfg.num_layers,
								 generator=lte,
								 label_pe=cfg.label_pe).to(cfg.device)
	else:
		model = UniversalTransformer(
			d_model=cfg.d_model,
			num_heads=cfg.num_heads,
			num_layers=cfg.num_layers,
			generator=lte,
			label_pe=cfg.label_pe,
			deterministic=cfg.deterministic,
		).to(cfg.device)
	model.load_state_dict(
		torch.load(
			os.path.join(hydra.utils.get_original_cwd(),
				f'../models/checkpoints/{cfg.ckpt}'), map_location=cfg.device)['ut_state_dict'])
	return model


def check_output_shape(output, Y, generator):
	if output.size() != Y[:, 1:].size():
		warn_str = f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing."
		warnings.warn(warn_str)
		print(warn_str)
		return _fix_output_shape(output, Y[:, 1:], generator)
	else:
		return output


def test_ood(model, generator, dp_name, num_samples=10, max_dp_value=10, use_y=False, tf=False, generator_kwargs=None, plot_ax=None, plot_label=None, regr=False):
	accuracy_values = []
	accuracy_std_values = []
	seq_acc_values = []
	seq_acc_std_values = []
	huber_loss_values = []
	dp_values = []  # dp = distribution parameter
	
	for dp_value in range(1, max_dp_value+1):
		logging.info(f"--- {dp_name} {dp_value} ---")
		same_nes_acc, same_nes_seq_acc = np.zeros(num_samples), np.zeros(num_samples)

		for sample_idx in range(num_samples):
			if dp_name.lower() == 'length':
				values = generator.generate_batch(dp_value, 1, **generator_kwargs)
			elif dp_name.lower() == 'nesting':
				values = generator.generate_batch(1, dp_value, **generator_kwargs)
			else:
				raise ValueError(f"Wrong distribution parameter: {dp_name}")
			
			if isinstance(generator, LTEStepsGenerator):
				X, Y, lenX, lenY, mask = values
			else:
				X, Y, lenX, lenY = values
			
			with torch.no_grad():
				model.eval()
				Y_model = Y[:, :-1] if use_y else None
				output = model(X, Y=Y_model, tf=tf)
				lenY = torch.tensor(lenY, device=X.device)

				if isinstance(model, UTwRegressionHead):
					classification_outputs, regression_outputs = output
					classification_outputs = check_output_shape(classification_outputs, Y, generator)
					avg_acc, _ = batch_acc(classification_outputs, Y[:, 1:], Y.size(-1), generator)
					regression_loss = torch.nn.functional.huber_loss(regression_outputs.squeeze(), get_mins_maxs_from_mask(mask))
					huber_loss_values += [regression_loss.item()]
				else:
					if dp_value <= 2:
						logging.info('\n'.join([f"{x} → {o}" for x, o in zip(generator.x_to_str(X), generator.y_to_str(output))]))
					output = check_output_shape(output, Y, generator)
					avg_acc, std_acc = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
					seq_acc_avg, seq_acc_std = batch_seq_acc(output, Y[:, 1:], generator, lenY)
					same_nes_acc[sample_idx] = avg_acc.item()
					same_nes_seq_acc[sample_idx] = seq_acc_avg.item()
		accuracy_values += [same_nes_acc.mean()]
		seq_acc_values += [same_nes_seq_acc.mean()]
		accuracy_std_values += [same_nes_acc.std()]
		seq_acc_std_values += [same_nes_seq_acc.std()]
		dp_values += [dp_value]
	
	df = pd.DataFrame()
	if regr:
		y_axis = 'Huber Loss'
		df[y_axis] = huber_loss_values
	else:
		y_axis = 'Character Accuracy'
		df[y_axis] = accuracy_values
		df['Character Accuracy Std'] = accuracy_std_values
		df['Sequence Accuracy'] = seq_acc_values
		df['Sequence Accuracy Std'] = seq_acc_std_values
		
	df[dp_name] = dp_values
	
	ax = sns.barplot(data=df, x=dp_name, y=y_axis, label=plot_label, ax=plot_ax, color='tab:blue')
	return ax, df


def plot_sample_attn_matrix(sample, attn_matrix, ax):
	cut_attn_matrix = attn_matrix[:len(sample), :len(sample)]
	# norm_cut_attn_matrix = (cut_attn_matrix - cut_attn_matrix.mean(axis=1))/cut_attn_matrix.std(axis=1)
	return sns.heatmap(data=cut_attn_matrix, xticklabels=sample, yticklabels=sample, ax=ax)


def plot_attn(model, generator, generator_kwargs):
	for n in range(1, 6):
		X, Y, _,_,_ = generator.generate_batch(2, n, **generator_kwargs)
		first_two_xs = X[:2]
		pred = model(first_two_xs)
		first_two_xs_str = generator.x_to_str(first_two_xs)
		first_two_xs_str = [x.replace('#', '') for x in first_two_xs_str]
		
		for idx, sample in enumerate(first_two_xs_str):
			fig, ax = plt.subplots(1, 1, figsize=(8, 7))
			attn_matrix = model.encoder.self_attn[idx].cpu().detach().numpy()
			ax = plot_sample_attn_matrix(sample, attn_matrix, ax=ax)
			wandb.log({f'N={n}': wandb.Image(fig)})


if __name__ == '__main__':
	main()
