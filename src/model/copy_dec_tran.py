import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from model.ut.UniversalTransformer import UniversalTransformer, UTDecoder
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
			X = self.encoder(X, src_mask)
			Y_pred_v = Y[:, 0, :].unsqueeze(1)
			output = Y_pred_v
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_pred_v)
				Y_pred = self.decoder(X, Y_pred, src_mask, None)
				Y_pred = self._copy_dec(X_1hot, X, Y_pred)
				Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
				pred_idx = Y_pred.argmax(-1)
				output = torch.concat([output, Y_pred], dim=1) 
				Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
				Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
			return output[:, 1:, :]  # cut SOS
		else:
			Y = self.y_emb(Y)
			X = self.encoder(X, src_mask)
			Y = self.decoder(X, Y, src_mask, tgt_mask)
			return self._copy_dec(X_1hot, X, Y)

	def _copy_dec(self, X_1hot, enc_emb, dec_emb):
		copy_q, copy_k, copy_v = self.linear_q(dec_emb), enc_emb, X_1hot
		p_2, attn_wts = self.copy_mha(copy_q, copy_k, copy_v)
		p_1 = self.final_proj(dec_emb)
		w = torch.sigmoid(self.linear_w(dec_emb))
		return w*p_1 + (1-w)*p_2


class CopyTransformer(UniversalTransformer):

	def __init__(self, d_model, num_heads, num_layers, generator, label_pe=False, device='cuda'):
		super().__init__(d_model, num_heads, num_layers, generator, label_pe=label_pe, device=device)
		self.decoder = CopyDecoder(d_model, num_heads, num_layers, vocab_dim=len(generator.x_vocab), label_pe=label_pe, device=device)

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
			X = self.encoder(X, src_mask)
			Y_pred_v = Y[:, 0, :].unsqueeze(1)
			output = Y_pred_v
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_pred_v)
				Y_pred = self.decoder(X, X_1hot, Y_pred, src_mask, None)
				Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
				pred_idx = Y_pred.argmax(-1)
				output = torch.concat([output, Y_pred], dim=1) 
				Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
				Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
			return output[:, 1:, :]  # cut SOS
		else:
			Y = self.y_emb(Y)
			X = self.encoder(X, src_mask)
			return self.decoder(X, X_1hot, Y, src_mask, tgt_mask)


class CopyDecoder(UTDecoder):

	def __init__(self, d_model, num_heads, num_layers, vocab_dim, dropout=0.1, label_pe=False, device='cpu'):
		super().__init__(d_model, num_heads, num_layers, dropout=dropout, label_pe=label_pe, device=device)
		self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, kdim=vocab_dim, vdim=vocab_dim)

	def forward(self, X, X1hot, Y, src_mask, tgt_mask):
		for l in range(self.num_layers):
			Y = self._pe(Y, self.label_pe)
			Y = self._decoder(X, X1hot, Y, src_mask, tgt_mask)
		return Y

	def _decoder(self, X, X1hot, Y, src_mask, tgt_mask):
		Yt, attn = self.MHSA(Y, X1hot, X1hot, key_padding_mask=src_mask)
		Y = Y + self.dropout1(Yt)
		Y = self.layer_norm1(Y)
		Yt, attn = self.MHA(Y, X, X, key_padding_mask=src_mask)
		Y = Y + self.dropout2(Yt)
		Y = self.layer_norm2(Y)
		Y = self.dropout3(self.transition_fn(Y))
		Y = self.layer_norm3(Y)
		return Y
