import torch
from model.ut.UniversalTransformer import UniversalTransformer
from pdb import set_trace


class CopyDecTran(UniversalTransformer):

	def __init__(self, d_model, num_heads, num_layers, generator, label_pe=False, device='cuda'):
		super().__init__(d_model, num_heads, num_layers, generator, label_pe=label_pe, device=device)
		self.copy_mha = torch.nn.MultiheadAttention(embed_dim=len(self.generator.x_vocab), num_heads=1,
													batch_first=True, kdim=d_model,
													vdim=len(self.generator.x_vocab))
		self.linear_q = torch.nn.Linear(d_model, len(self.generator.x_vocab))
		self.linear_w = torch.nn.Linear(d_model, 1)

	def forward(self, X, Y=None, tf=False):
		if Y is not None:
			return self._fwd(X, Y, tf=tf)
		else:
			pass

	def _fwd(self, X, Y, tf=False):
		src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
		tgt_mask = (Y.argmax(-1) == self.generator.y_vocab['#'])
		X_1hot = X
		X = self.x_emb(X)

		if not tf:
			pass
			if self.act_enc is None:
				X = self.encoder(X, src_mask)
			else:
				X, _ = self.encoder(X, src_mask)
			Y_pred_v = Y[:, 0, :].unsqueeze(1)
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_pred_v)
				Y_pred = self.decoder(X, Y_pred, src_mask, None)
				Y_pred = self.final_proj(Y_pred)
				Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
				pred_idx = Y_pred.argmax(-1) 
				Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
				Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
				return Y_pred_v
		else:
			Y = self.y_emb(Y)
			X = self.encoder(X, src_mask)
			Y = self.decoder(X, Y, src_mask, tgt_mask)

			copy_q, copy_k, copy_v = self.linear_q(Y), X, X_1hot
			p_2, attn_wts = self.copy_mha(copy_q, copy_k, copy_v)
			p_1 = self.final_proj(Y)
			w = torch.sigmoid(self.linear_w(Y))
			return w*p_1 + (1-w)*p_2
