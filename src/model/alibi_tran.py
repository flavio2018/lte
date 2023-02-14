import torch
import torch.nn.functional as F
import math
from model.ut.UniversalTransformer import UniversalTransformer, UTEncoder, UTDecoder, _gen_bias_mask
from pdb import set_trace


def get_slopes(n):
	"""Code taken from original paper repo: https://github.com/ofirpress/attention_with_linear_biases/issues/5"""
	def get_slopes_power_of_2(n):
		start = (2**(-2**-(math.log2(n)-3)))
		ratio = start
		return [start*ratio**i for i in range(n)]

	if math.log2(n).is_integer():
		return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
	else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
		closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
		return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class AlibiTran(UniversalTransformer):
	def __init__(self, d_model, num_heads, num_layers, generator, dropout=0.1, device='cuda'):
		super().__init__(d_model, num_heads, num_layers, generator, dropout=dropout, device=device)
		self.encoder = AlibiEncoder(d_model, num_heads, num_layers, dropout=dropout, device=device)
		self.decoder = AlibiDecoder(d_model, num_heads, num_layers, dropout=dropout, device=device)
		

	def forward(self, X, Y=None, tf=False):
		if Y is not None:
			return self._fwd(X, Y, tf=tf)
		else:
			pass

	def _fwd(self, X, Y, tf=False):
		src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
		tgt_mask = (Y.argmax(-1) == self.generator.y_vocab['#'])
		X = self.x_emb(X)
		
		if not tf:
			X = self.encoder(X, src_mask)
			Y_pred_v = Y[:, 0, :].unsqueeze(1)
			output = Y_pred_v
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_pred_v)
				Y_pred = self.decoder(X, Y_pred, src_mask, None)
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
			return self.final_proj(Y)

class AlibiEncoder(UTEncoder):

	def __init__(self, d_model, num_heads, num_layers, dropout=0.1, device='cpu'):
		super().__init__(d_model, num_heads, num_layers, dropout=dropout, device=device)

	def forward(self, X, src_mask):
		for l in range(self.num_layers):  # we use no timestep encoding (as in UT) so the model cannot distinguish between different depths
			X = self._encoder(X, src_mask)
		return X

	def _encoder(self, X, src_mask):
		Xt, attn = self.MHSA(X, X, X, key_padding_mask=src_mask, attn_mask=self._alibi_attn_weights(X))
		X = X + self.dropout1(Xt)
		X = self.layer_norm1(X)
		X = X + self.dropout2(self.transition_fn(X))
		X = self.layer_norm2(X)
		return X

	def _alibi_attn_weights(self, X):
		"""Code adapted from original paper repo: https://github.com/ofirpress/attention_with_linear_biases/issues/5"""
		maxpos = 5000  # maximum expected tokens per sample, as in LPE
		attn_heads = self.MHSA.num_heads

		context_position = torch.arange(maxpos, device=X.device)[:, None]
		memory_position = torch.arange(maxpos, device=X.device)[None, :]
		relative_position = memory_position - context_position 
		relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)

		slopes = torch.Tensor(get_slopes(attn_heads)).cuda()*-1
		alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position

		src_len = X.size(1)
		bs = X.size(0)

		return (alibi[:, :src_len, :src_len]
					.tile(bs, 1, 1, 1)
					.view(bs*attn_heads, src_len, src_len))



class AlibiDecoder(UTDecoder):

	def __init__(self, d_model, num_heads, num_layers, dropout=0.1, device='cpu'):
		super().__init__(d_model, num_heads, num_layers, dropout=dropout, device=device)

	def forward(self, X, Y, src_mask, tgt_mask):
		for l in range(self.num_layers):
			Y = self._decoder(X, Y, src_mask, tgt_mask)
		return Y

	def _decoder(self, X, Y, src_mask, tgt_mask):
		Yt, attn = self.MHSA(Y, Y, Y, attn_mask=self._alibi_attn_weights(Y), key_padding_mask=tgt_mask)
		Y = Y + self.dropout1(Yt)
		Y = self.layer_norm1(Y)
		Yt, attn = self.MHA(Y, X, X, key_padding_mask=src_mask)
		Y = Y + self.dropout2(Yt)
		Y = self.layer_norm2(Y)
		Y = self.dropout3(self.transition_fn(Y))
		Y = self.layer_norm3(Y)
		return Y

	def _alibi_attn_weights(self, Y):
		"""Code adapted from original paper repo: 
		https://github.com/ofirpress/attention_with_linear_biases/blob/02aa87e7a29e9340efd28d6d169018eafb3aa57a/fairseq/models/transformer.py#L760"""
		maxpos = 5000  # maximum expected tokens per sample, as in LPE
		attn_heads = self.MHA.num_heads

		bias_mask = _gen_bias_mask(Y.size(1), Y.device).tile(Y.size(0)*attn_heads, 1, 1)
		bias_mask = torch.where(bias_mask, -torch.inf, 0).to(Y.device)

		slopes = torch.tensor(get_slopes(attn_heads), device=Y.device)
		alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos, device=Y.device).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
		alibi = alibi.view(attn_heads, 1, maxpos)
		bs, tgt_len = Y.size(0), Y.size(1)
		alibi = alibi.repeat(bs, 1, 1)  # batch_size, 1, 1
		alibi_dec_mha = alibi[:, :, :tgt_len]
		return bias_mask + alibi_dec_mha
