import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from model.ut.UniversalTransformer import UniversalTransformer, UTDecoder, UTEncoder, _gen_timing_signal
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
				Y_pred = self.final_proj(Y_pred)
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

	def _test_fwd(self, X):
		pass


class CopyTransformer(UniversalTransformer):

	def __init__(self, d_model, num_heads, num_layers, generator, label_pe=False, device='cuda'):
		super().__init__(d_model, num_heads, num_layers, generator, label_pe=label_pe, device=device)
		self.encoder = CopyEncoder(d_model, num_heads, num_layers, vocab_dim=len(generator.x_vocab), label_pe=label_pe, device=device)
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
		X_proj = self.x_emb(X)

		if not tf:
			X_emb = self.encoder(X_proj, X_1hot, src_mask)
			Y_pred_v = Y[:, 0, :].unsqueeze(1)
			output = Y_pred_v
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_pred_v)
				Y_pred = self.decoder(X_emb, X_1hot, Y_pred, src_mask, None)
				Y_pred = self.final_proj(Y_pred)
				Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
				pred_idx = Y_pred.argmax(-1)
				output = torch.concat([output, Y_pred], dim=1) 
				Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
				Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
			return output[:, 1:, :]  # cut SOS
		else:
			Y = self.y_emb(Y)
			X_emb = self.encoder(X_proj, X_1hot, src_mask)
			Y_emb = self.decoder(X_emb, X_1hot, Y, src_mask, tgt_mask)
			return self.final_proj(Y_emb)

	def _test_fwd(self, X):
		it, max_it = 0, 100
		src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
		X_1hot = X
		X_proj = self.x_emb(X)
		EOS_idx = self.generator.y_vocab['.']
		X_emb = self.encoder(X_proj, X_1hot, src_mask)
		stopped = torch.zeros(X.size(0)).type(torch.BoolTensor).to(X.device)
		Y_pred_v = torch.tile(F.one_hot(torch.tensor([self.generator.y_vocab['?']]), num_classes=len(self.generator.y_vocab)), dims=(X.size(0), 1, 1)).type(torch.FloatTensor).to(X.device)

		while not stopped.all() and (it < max_it):
			it += 1
			Y_pred = self.y_emb(Y_pred_v)
			Y_pred = self.decoder(X_emb, X_1hot, Y_pred, src_mask, None)
			Y_pred = self.final_proj(Y_pred)
			Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
			pred_idx = Y_pred.argmax(-1)  # convert to indices
			Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)  # make 1hot
			Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)  # combine w/ previous outputs
			stopped = torch.logical_or((pred_idx.squeeze() == EOS_idx), stopped)
		return Y_pred_v[:, 1:, :]  # cut SOS


class CopyDecoder(UTDecoder):

	def __init__(self, d_model, num_heads, num_layers, vocab_dim, dropout=0.1, label_pe=False, device='cpu'):
		super().__init__(d_model, num_heads, num_layers, dropout=dropout, label_pe=label_pe, device=device)
		self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, kdim=vocab_dim, vdim=vocab_dim)
		positional_encoding_vocab = _gen_timing_signal(5000, vocab_dim)
		self.register_buffer('positional_encoding_vocab', positional_encoding_vocab)

	def forward(self, X, X_1hot, Y, src_mask, tgt_mask):
		for l in range(self.num_layers):
			Y = self._pe(Y, self.label_pe)
			X_1hot = self._pe(X_1hot, self.label_pe, vocab=True)
			Y = self._decoder(X, X_1hot, Y, src_mask, tgt_mask)
		return Y

	def _decoder(self, X, X_1hot, Y, src_mask, tgt_mask):
		Yt, attn = self.MHSA(Y, X_1hot, X_1hot, key_padding_mask=src_mask)
		Y = Y + self.dropout1(Yt)
		Y = self.layer_norm1(Y)
		Yt, attn = self.MHA(Y, X, X, key_padding_mask=src_mask)
		Y = Y + self.dropout2(Yt)
		Y = self.layer_norm2(Y)
		Y = self.dropout3(self.transition_fn(Y))
		Y = self.layer_norm3(Y)
		return Y

	def _pe(self, X, label=False, vocab=False):
		positional_encoding = self.positional_encoding_vocab if vocab else self.positional_encoding
		if label:
			max_seq_len = X.size(1)
			max_pe_pos = self.positional_encoding.size(1)
			val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
			return X + self.dropout1(positional_encoding[:, val, :])
		else:
			return X + self.dropout1(positional_encoding[:, :X.size(1), :])


class CopyEncoder(UTEncoder):

	def __init__(self, d_model, num_heads, num_layers, vocab_dim, dropout=0.1, label_pe=False, device='cpu'):
		super().__init__(d_model, num_heads, num_layers, dropout=dropout, label_pe=label_pe, device=device)
		self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, kdim=vocab_dim, vdim=vocab_dim)
		positional_encoding_vocab = _gen_timing_signal(5000, vocab_dim)
		self.register_buffer('positional_encoding_vocab', positional_encoding_vocab)

	def forward(self, X, X_1hot, src_mask):
		for l in range(self.num_layers):
			X = self._pe(X, self.label_pe)
			X_1hot = self._pe(X_1hot, self.label_pe, vocab=True)
			X = self._encoder(X, X_1hot, src_mask)
		return X

	def _encoder(self, X, X_1hot, src_mask):
		Xt, attn = self.MHSA(X, X_1hot, X_1hot, key_padding_mask=src_mask)
		X = X + self.dropout1(Xt)
		X = self.layer_norm1(X)
		X = X + self.dropout2(self.transition_fn(X))
		X = self.layer_norm2(X)
		return X

	def _pe(self, X, label=False, vocab=False):
		positional_encoding = self.positional_encoding_vocab if vocab else self.positional_encoding
		if label:
			max_seq_len = X.size(1)
			max_pe_pos = self.positional_encoding.size(1)
			val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
			return X + self.dropout1(positional_encoding[:, val, :])
		else:
			return X + self.dropout1(positional_encoding[:, :X.size(1), :])
