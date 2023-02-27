from data.generator import LTEGenerator
from model.ut.UniversalTransformer import UniversalTransformer
from model.test import batch_acc, _fix_output_shape
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from icecream import ic


@hydra.main(config_path="../../conf/local", config_name="test_e2e", version_base='1.2')
def main(cfg):
	warnings.filterwarnings('always', category=UserWarning)
	ic.enable() if cfg.enable_ic else ic.disable()

	lte = LTEGenerator(cfg.device)

	ut_e2e = UniversalTransformer(
	  d_model=cfg.d_model,
	  num_heads=cfg.num_heads,
	  num_layers=cfg.num_layers,
	  generator=lte,
	  label_pe=cfg.label_pe,
	).to(cfg.device)
	ut_e2e.load_state_dict(
		torch.load(
			os.path.join(hydra.utils.get_original_cwd(),
				f'../models/checkpoints/{cfg.ckpt}'), map_location=cfg.device)['ut_state_dict'])

	ax = test_ood(ut_e2e, lte, 'nesting', use_y=cfg.use_y,
					generator_kwargs={'split': 'test',
									  'batch_size': cfg.bs,
									  'ops': cfg.ops})
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.ckpt[:-4]}_end2end.pdf"))
	

def test_ood(model, generator, dp_name, max_dp_value=10, use_y=False, tf=False, generator_kwargs=None, plot_ax=None, plot_label=None):
	accuracy_values = []
	dp_values = []  # dp = distribution parameter
	
	for dp_value in range(1, max_dp_value+1):
		ic("nesting", dp_value)
		if dp_name == 'length':
			values = generator.generate_batch(dp_value, 1, **generator_kwargs)
		elif dp_name == 'nesting':
			values = generator.generate_batch(1, dp_value, **generator_kwargs)
		else:
			raise ValueError(f"Wrong distribution parameter: {dp_name}")

		if isinstance(generator, LTEStepsGenerator):
			X, Y, lenX, lenY, _ = values
		else:
			X, Y, lenX, lenY = values

		with torch.no_grad():
			model.eval()
			Y_model = Y[:, :-1] if use_y else None
			output = model(X, Y=Y_model, tf=tf)
			if output.size() != Y[:, 1:].size():
				warn_str = f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing."
				warnings.warn(warn_str)
				ic(warn_str)
				output = _fix_output_shape(output, Y[:, 1:], generator)

			acc = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
			accuracy_values += [acc.item()]
			dp_values += [dp_value]
	
	df = pd.DataFrame()
	df['Character Accuracy'] = accuracy_values
	df[dp_name] = dp_values
	
	ax = sns.barplot(data=df, x=dp_name, y='Character Accuracy', label=plot_label, ax=plot_ax, color='tab:blue')
	if plot_ax is not None:
		ax.figure
	return ax


if __name__ == '__main__':
	main()
