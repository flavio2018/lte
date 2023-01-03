import hydra
import omegaconf
from datetime import datetime as dt
import os
import torch
import numpy as np
from model.ut.UniversalTransformer import UniversalTransformer
from model.ut.ACT import ACT
from model.test import compute_loss, batch_acc, eval_ut, compute_act_loss
from data.generator import LTEGenerator
from tqdm import trange
import wandb


FREQ_EVAL = 10


@hydra.main(config_path="../conf/local", config_name="train_ut")
def train_ut(cfg):
	print(omegaconf.OmegaConf.to_yaml(cfg))

	lte = LTEGenerator(cfg.device)
	ut = UniversalTransformer(d_model=cfg.d_model,
							  num_heads=cfg.num_heads,
							  num_layers=cfg.num_layers,
							  generator=lte,
							  act_enc=ACT(d_model=cfg.d_model,
							  	  		  max_hop=cfg.num_layers),
							  act_dec=ACT(d_model=cfg.d_model,
							  	  		  max_hop=cfg.num_layers),
							  device=cfg.device).to(cfg.device)

	xent = torch.nn.CrossEntropyLoss(reduction='none')
	opt = torch.optim.Adam(ut.parameters(), lr=cfg.lr)
	FREQ_WANDB_LOG = np.ceil(cfg.max_iter / 100000)

	wandb.init(
		project="lte",
		entity="flapetr",
		mode="online",
		settings=wandb.Settings(start_method="fork"),
	)
	wandb.run.name = cfg.codename
	wandb.config.update(omegaconf.OmegaConf.to_container(
		cfg, resolve=True, throw_on_missing=True))
	wandb.watch(ut, log_freq=FREQ_WANDB_LOG)
	start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')

	for i_step in range(cfg.max_iter):
		X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops)
		loss_step, acc_step, act = train_step(ut, (X_1h, Y_1h), lte, xent, opt)
		R_enc, N_enc, R_dec, N_dec = act

		X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops, split='valid')
		loss_valid_step, acc_valid_step, _ = valid_step(ut, (X_1h, Y_1h), lte, xent)
		
		if i_step % FREQ_WANDB_LOG == 0:
			wandb.log({
					"loss": loss_step,
					"acc": acc_step,
					"val_loss": loss_valid_step,
					"val_acc": acc_valid_step,
					"avg_n_updates_enc": N_enc.mean().item(),
					"avg_n_updates_dec": N_dec.mean().item(),
					"update": i_step,
				})
		
		if i_step % 1000 == 0:
			torch.save({
					'update': i_step,
					'ut_state_dict': ut.state_dict(),
					'opt': opt.state_dict(),
					'loss_train': loss_step,
				}, os.path.join(hydra.utils.get_original_cwd(), f"../models/checkpoints/{start_timestamp}_{cfg.codename}.pth"))
			eval_ut(ut, X_1h, Y_1h, xent, lte, cfg.device)

		if i_step % FREQ_EVAL == 0:
			X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops, split='valid')
			_, acc_step_test, act = valid_step(ut, (X_1h, Y_1h), lte, xent)
			R_enc, N_enc, R_dec, N_dec = act
			wandb.log({
				"acc_test": acc_step_test,
				"avg_n_updates_enc_test": N_enc.mean().item(),
				"avg_n_updates_dec_test": N_dec.mean().item(),
				"test_update": i_step // FREQ_EVAL,
			})
	

def train_step(model, data, generator, loss, opt):
	opt.zero_grad()
	model.train()
	inputs, targets = data
	
	outputs, act = model(inputs, targets[:, :-1])
	avg_loss = compute_loss(loss, [outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], generator)
	avg_loss += 0.01*compute_act_loss(outputs, act, inputs, targets[:, 1:], generator)
	avg_acc = batch_acc([outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], targets.size(-1), generator)
	
	avg_loss.backward()
	opt.step()
	return avg_loss.item(), avg_acc.item(), act


def valid_step(model, data, generator, loss):
	model.eval()
	inputs, targets = data
	
	outputs, act = model(inputs, targets[:, :-1])
	avg_loss = compute_loss(loss, [outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], generator)
	avg_loss += 0.01*compute_act_loss(outputs, act, inputs, targets[:, 1:], generator)
	avg_acc = batch_acc([outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], targets.size(-1), generator)
	return avg_loss.item(), avg_acc.item(), act


if __name__ == '__main__':
	train_ut()
