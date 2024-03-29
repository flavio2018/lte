import hydra
import omegaconf
from datetime import datetime as dt
import os
import torch
import numpy as np
from model.ut.UniversalTransformer import UniversalTransformer
from model.ut.ACT import ACT
from model.copy_dec_tran import CopyDecTran, CopyTransformer
from model.alibi_tran import AlibiTran
from model.regression_tran import UTwRegressionHead
from model.test import compute_loss, batch_acc, compute_act_loss
from data.generator import LTEGenerator, LTEStepsGenerator, get_mins_maxs_from_mask
import wandb


@hydra.main(config_path="../conf/local", config_name="train_ood")
def train_ood(cfg):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    if cfg.step_generator:
        lte = LTEStepsGenerator(cfg.device, cfg.same_vocab, cfg.hash_split)
        lte_kwargs = {
            "batch_size": cfg.bs,
            "simplify": cfg.simplify,
            "simplify_w_value": cfg.simplify_w_value,
            "filtered_swv": cfg.filtered_swv,
            "filtered_s2e": cfg.filtered_s2e,
            "substitute": cfg.substitute,
            "split": "train",
        }
        lte.load_sample2split(hydra.utils.get_original_cwd())
    else:
        lte = LTEGenerator(cfg.device)
        lte_kwargs = {
            "batch_size": cfg.bs,
            "split": "train",
        }

    if cfg.copy_dec:
        model = CopyDecTran(d_model=cfg.d_model,
                        num_heads=cfg.num_heads,
                        num_layers=cfg.num_layers,
                        generator=lte,
                        label_pe=cfg.label_pe).to(cfg.device)
    elif cfg.alibi:
        model = AlibiTran(d_model=cfg.d_model,
                        num_heads=cfg.num_heads,
                        num_layers=cfg.num_layers,
                        generator=lte).to(cfg.device)
    elif cfg.copy_ut:
        model = CopyTransformer(d_model=cfg.d_model,
                             num_heads=cfg.num_heads,
                             num_layers=cfg.num_layers,
                             generator=lte,
                             label_pe=cfg.label_pe).to(cfg.device)
    elif cfg.regr_ut:
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
        #     act_enc=ACT(d_model=d_model,
        #                 max_hop=num_layers),
        #     act_dec=ACT(d_model=d_model,
        #                 max_hop=num_layers),
            label_pe=cfg.label_pe,
        ).to(cfg.device)
    xent = torch.nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.ckpt:
        ckpt = torch.load(
            os.path.join(hydra.utils.get_original_cwd(),
                f'../models/checkpoints/{cfg.ckpt}'), map_location=cfg.device)
        model.load_state_dict(ckpt['ut_state_dict'])
        opt.load_state_dict(ckpt['opt'])

    FREQ_WANDB_LOG = np.ceil(cfg.max_iter / 1000)
    wandb.init(
        project="lte",
        entity="flapetr",
        mode="online",
        settings=wandb.Settings(start_method="fork"))
    wandb.run.name = cfg.codename
    wandb.config.update(omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model, log_freq=FREQ_WANDB_LOG)
    start_timestamp = dt.now().strftime('%Y-%m-%d_%H-%M')

    for it in range(cfg.max_iter):
        lte_kwargs['split']='train'
        if isinstance(cfg.tf, float):
            tf = True if (torch.rand(1) > cfg.tf).item() else False
        else:
            tf = cfg.tf

        loss_step, acc_step = train_step(model, lte, cfg.max_len, cfg.max_nes, lte_kwargs, opt, xent, tf=tf)

        if it % FREQ_WANDB_LOG == 0:
            lte_kwargs['split']='valid'

            loss_valid_step, acc_valid_step = valid_step(model, lte, cfg.max_len, cfg.max_nes, lte_kwargs, xent, tf=cfg.tf)
            lte_kwargs['split']='test'
            
            loss_ood_nes, acc_ood_nes = valid_step(model, lte, cfg.max_len, cfg.max_nes+2, lte_kwargs, xent, tf=cfg.tf)

            wandb.log({
                    "loss": loss_step,
                    "acc": acc_step,
                    "val_loss": loss_valid_step,
                    "val_acc": acc_valid_step,
                    "ood_nes_acc": acc_ood_nes,
                    "update": it,
                })    

        if it % 1000 == 0:
            torch.save({
                    'update': it,
                    'ut_state_dict': model.state_dict(),
                    'opt': opt.state_dict(),
                    'loss_train': loss_step,
                }, os.path.join(hydra.utils.get_original_cwd(), f"../models/checkpoints/{start_timestamp}_{cfg.codename}.pth"))
            

def train_step(model, lte, max_length, max_nesting, lte_kwargs, opt, xent, tf=False):
    model.train()
    opt.zero_grad()

    if isinstance(lte, LTEStepsGenerator):
        X, Y, lenX, lenY, mask = lte.generate_batch(max_length, max_nesting, **lte_kwargs)
    else:
        X, Y, lenX, lenY = lte.generate_batch(max_length, max_nesting, **lte_kwargs)

    outputs = model(X, Y[:, :-1], tf=tf)    
    if isinstance(model, UTwRegressionHead):
        classification_outputs, regression_outputs = outputs
        classification_loss = compute_loss(xent, classification_outputs, Y[:, 1:], lte)
        acc, _ = batch_acc(classification_outputs, Y[:, 1:], Y.size(-1), lte)
        regression_loss = torch.nn.functional.huber_loss(regression_outputs.squeeze(), get_mins_maxs_from_mask(mask))
        loss = classification_loss + regression_loss
    else:
        loss = compute_loss(xent, outputs, Y[:, 1:], lte)
        acc, _ = batch_acc(outputs, Y[:, 1:], Y.size(-1), lte)

    loss.backward()
    opt.step()
    return loss.item(), acc.item()


def valid_step(model, lte, max_length, max_nesting, lte_kwargs, xent, tf=False):
    model.eval()

    if isinstance(lte, LTEStepsGenerator):
        X, Y, lenX, lenY, mask = lte.generate_batch(max_length, max_nesting, **lte_kwargs)
    else:
        X, Y, lenX, lenY = lte.generate_batch(max_length, max_nesting, **lte_kwargs)
    
    outputs = model(X, Y[:, :-1], tf=tf)    
    if isinstance(model, UTwRegressionHead):
        classification_outputs, regression_outputs = outputs
        classification_loss = compute_loss(xent, classification_outputs, Y[:, 1:], lte)
        acc, _ = batch_acc(classification_outputs, Y[:, 1:], Y.size(-1), lte)
        regression_loss = torch.nn.functional.huber_loss(regression_outputs.squeeze(), get_mins_maxs_from_mask(mask))
        loss = classification_loss + regression_loss
    else:
        loss = compute_loss(xent, outputs, Y[:, 1:], lte)
        acc, _ = batch_acc(outputs, Y[:, 1:], Y.size(-1), lte)
    return loss.item(), acc.item()
    

if __name__ == '__main__':
    train_ood()
