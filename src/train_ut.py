import torch
import numpy as np
from model.ut.UniversalTransformer import UniversalTransformer
from model.test import compute_loss, batch_acc
from data.generator import LTEGenerator
import wandb


FREQ_EVAL = 10


def train_ut(cfg):
	print(omegaconf.OmegaConf.to_yaml(cfg))

	lte = LTEGenerator(cfg.device)
	ut = UniversalTransformer(d_model=cfg.d_model,
							  num_heads=cfg.num_heads,
							  num_layers=cfg.num_layers,
							  x_vocab_size=len(lte.x_vocab),
							  y_vocab_size=len(lte.y_vocab),
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
	start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')

	for i_step in range(cfg.max_iter):
		X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops)
		loss_step, acc_step = train_step(ut, (X_1h, Y_1h), lte, xent, opt)
		
		X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops, split='valid')
		loss_valid_step, acc_valid_step = valid_step(ut, (X_1h, Y_1h), lte, xent)
		
		if i_step % FREQ_WANDB_LOG == 0:
			wandb.log({
					"loss": loss_step,
					"acc": acc_step,
					"val_loss": loss_valid_step,
					"val_acc": acc_valid_step,
					"update": i_step,
				})
		
		if i_step % 1000 == 0:
			torch.save({
					'update': i_step,
					'ut_state_dict': ut.state_dict(),
					'opt': opt.state_dict(),
					'loss_train': loss_val,
				}, os.path.join(hydra.utils.get_original_cwd(), f"../models/checkpoints/{start_timestamp}_{cfg.codename}.pth"))

		if i_step % FREQ_EVAL == 0:
			X_1h, Y_1h, len_x, len_y = lte.generate_batch(cfg.max_len, cfg.max_nes, cfg.bs, ops=cfg.ops, split='valid')
			_, acc_step_test = valid_step(ut, (X_1h, Y_1h), lte, xent)
			wandb.log({
				"acc_test": acc_step_test,
				"test_update": i_step // FREQ_EVAL,
			})
	

def train_step(model, data, generator, loss, opt):
	opt.zero_grad()
	model.train()
	inputs, targets = data
	
	outputs = model(inputs, targets[:, :-1])
	avg_loss = compute_loss(loss, [outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], generator)
	avg_acc = batch_acc([outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], targets.size(-1), generator)
	
	avg_loss.backward()
	opt.step()
	return avg_loss.item(), avg_acc.item()


def valid_step(model, data, generator, loss):
	model.eval()
	inputs, targets = data
	
	outputs = model(inputs, targets[:, :-1])
	avg_loss = compute_loss(loss, [outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], generator)
	avg_acc = batch_acc([outputs[:, pos, :] for pos in range(outputs.size(1))], targets[:, 1:], targets.size(-1), generator)
	return avg_loss.item(), avg_acc.item()


if __name__ == '__main__':
	train_ut()
