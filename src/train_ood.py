import hydra
import omegaconf
from datetime import datetime as dt
import os
import torch
import numpy as np
from model.ut.UniversalTransformer import UniversalTransformer
from model.ut.ACT import ACT
from model.test import compute_loss, batch_acc, compute_act_loss
from data.generator import LTEStepsGenerator
import wandb


@hydra.main(config_path="../conf/local", config_name="train_ood")
def train_ood(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    lte_step = LTEStepsGenerator(cfg.device)

    ut = UniversalTransformer(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        generator=lte_step,
    #     act_enc=ACT(d_model=d_model,
    #                 max_hop=num_layers),
    #     act_dec=ACT(d_model=d_model,
    #                 max_hop=num_layers),
    ).to(cfg.device)
    xent = torch.nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.Adam(ut.parameters(), lr=cfg.lr)
    FREQ_WANDB_LOG = np.ceil(cfg.max_iter / 100000)
    wandb.init(
        project="lte",
        entity="flapetr",
        mode="online",
        settings=wandb.Settings(start_method="fork"))
    wandb.run.name = cfg.codename
    wandb.config.update(omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))
    wandb.watch(ut, log_freq=FREQ_WANDB_LOG)
    start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')

    for it in range(cfg.max_iter):
        loss_step, acc_step = train_step(ut, lte_step, cfg.max_len, cfg.max_nes, cfg.bs, opt, xent, masked=cfg.masked)
        loss_valid_step, acc_valid_step = valid_step(ut, lte_step, cfg.max_len, cfg.max_nes, cfg.bs, xent, masked=cfg.masked)

        if it % FREQ_WANDB_LOG == 0:
            wandb.log({
                    "loss": loss_step,
                    "acc": acc_step,
                    "val_loss": loss_valid_step,
                    "val_acc": acc_valid_step,
                    "update": it,
                })    

        if it % 1000 == 0:
            torch.save({
                    'update': it,
                    'ut_state_dict': ut.state_dict(),
                    'opt': opt.state_dict(),
                    'loss_train': loss_step,
                }, os.path.join(hydra.utils.get_original_cwd(), f"../models/checkpoints/{start_timestamp}_{cfg.codename}.pth"))
            

def train_step(model, lte, max_length, max_nesting, batch_size, opt, xent, masked=False):
    model.train()
    opt.zero_grad()
    mask = None

    X, Y, lenX, lenY, mask = lte.generate_batch(max_length, max_nesting, batch_size=batch_size)
    if not masked:
        mask = None
    x_idx = x.argmax(-1)
    padded_x = torch.where(~mask, x_idx, lte.x_vocab['#'])
    X = torch.nn.functional.one_hot(padded_x, num_classes=len(lte.x_vocab)).type(torch.float)
    
    outputs = model(X, Y[:, :-1], mask)
    loss = compute_loss(xent, outputs, Y[:, 1:], lte)
    # loss += 0.01*compute_act_loss(outputs, act, X, Y[:, 1:], lte)
    acc = batch_acc(outputs, Y[:, 1:], Y.size(-1), lte)

    loss.backward()
    opt.step()
    return loss.item(), acc.item()


def valid_step(model, lte, max_length, max_nesting, batch_size, xent, masked=False):
    model.eval()
    mask = None

    X, Y, lenX, lenY, mask = lte.generate_batch(max_length, max_nesting, batch_size=batch_size, split='valid')
    if not masked:
        mask = None
    
    outputs = model(X, Y[:, :-1], mask)
    loss = compute_loss(xent, outputs, Y[:, 1:], lte)
    # loss += 0.01*compute_act_loss(outputs, act, X, Y[:, 1:], lte)
    acc = batch_acc(outputs, Y[:, 1:], Y.size(-1), lte)
    return loss.item(), acc.item()
    

if __name__ == '__main__':
    train_ood()
