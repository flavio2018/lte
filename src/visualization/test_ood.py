import hydra
import omegaconf
import torch
import pandas as pd
import seaborn as sns
import datetime as dt
import os
from data.generator import LTEGenerator, LTEStepsGenerator, get_mins_maxs_from_mask
from model.regression_tran import UTwRegressionHead
from model.ut.UniversalTransformer import UniversalTransformer
from test import batch_acc


@hydra.main(config_path="../conf/local", config_name="test_ood")
def main():
    start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')
    model_id = 'regr_ut' if cfg.regr_ut else 'ut'
    task_id = 'simplify_w_value' if cfg.simplify_w_value else 'simplify'

	lte, lte_kwargs = build_generator(cfg)
	model = load_model(cfg, lte)
	metric = 'characc'
	ax = test_ood(model, lte, 'nesting', trials=cfg.num_trials, generator_kwargs=lte_kwargs)
	fig = ax.get_figure()
	fig.savefig(f"../../reports/figures/{start_timestamp}_{task_id}_{model_id}_{metric}.pdf")
	if isinstance(model, UTwRegressionHead):
		metric = 'huberloss'
		ax = test_ood(model, lte, 'nesting', trials=cfg.num_trials, generator_kwargs=lte_kwargs, regr=True)
		fig = ax.get_figure()
		fig.savefig(f"../../reports/figures/{start_timestamp}_{task_id}_{model_id}_{metric}.pdf")


def build_generator(cfg):
	if cfg.step_generator:
        lte = LTEStepsGenerator(cfg.device, cfg.same_vocab)
        lte_kwargs = {
            "batch_size": cfg.bs,
            "simplify_w_value": cfg.simplify_w_value,
            "split": "test",
        }
    else:
        lte = LTEGenerator(cfg.device)
        lte_kwargs = {
            "batch_size": cfg.bs,
            "split": "test",
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
        ).to(cfg.device)
    model.load_state_dict(
    	torch.load(
    		os.path.join(hydra.utils.get_original_cwd(),
    			f'../models/checkpoints/{cfg.ckpt}'), map_location=cfg.device)['ut_state_dict'])
	return model


def test_ood(model, generator, dp_name, max_dp_value=10, trials=10, generator_kwargs=None, plot_ax=None, plot_label=None, regr=False):
    accuracy_values = []
    huber_loss_values = []
    dp_values = []  # dp = distribution parameter
    
    for trial in range(trials):
        for dp_value in range(1, max_dp_value+1):
            if dp_name == 'length':
                values = generator.generate_batch(dp_value, 1, **generator_kwargs)
            elif dp_name == 'nesting':
                values = generator.generate_batch(1, dp_value, **generator_kwargs)
            else:
                raise ValueError(f"Wrong distribution parameter: {dp_name}")
            
            if isinstance(generator, LTEStepsGenerator):
                X, Y, lenX, lenY, mask = values
            else:
                X, Y, lenX, lenY = values
            
            with torch.no_grad():
                model.eval()
                output = model(X, Y=None, tf=False)
                if isinstance(model, UTwRegressionHead):
        	        classification_outputs, regression_outputs = outputs
				    acc = batch_acc(classification_outputs, Y[:, 1:], Y.size(-1), generator)
			        regression_loss = torch.nn.functional.huber_loss(regression_outputs[:, -1], get_mins_maxs_from_mask(mask))
			        huber_loss_values += [regression_loss]
                else:
	                acc = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
                accuracy_values += [acc.item()]
                dp_values += [dp_value]
    
    df = pd.DataFrame()
    if regr:
    	y_axis = 'Huber Loss'
    	df[y_axis] = huber_loss_values
    else:
    	y_axis = 'Character Accuracy'
	    df[y_axis] = accuracy_values
    df[dp_name] = dp_values
    
    ax = sns.lineplot(data=df, x=dp_name, y=y_axis, label=plot_label, ax=plot_ax)
    return ax
