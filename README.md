# DNMT
if parallel_apply assert len(modules) == len(inputs)
change torch/nn/parallel/distributed.py line 216: outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)