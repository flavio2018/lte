import torch
from torch.nn import Linear
from model.ut.UniversalTransformer import UniversalTransformer


class UTwRegressionHead(UniversalTransformer):

	def __init__(self, d_model, num_heads, num_layers, generator, label_pe=False, dropout=0.1, device='cuda'):
		super().__init__(d_model, num_heads, num_layers, generator, label_pe=label_pe, dropout=dropout, device=device)
		self.regression_head = Linear(d_model, 2)


	def forward(self, X, Y=None, tf=False):
		if Y is not None:
			classification_output = self._fwd(X, Y, tf=tf)
		else:
			classification_output = self._test_fwd(X)
		regression_output = self.regression_head(self.Y_logits)
		return classification_output, regression_output
