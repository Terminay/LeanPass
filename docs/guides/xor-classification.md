---
sidebar_position: 2
---

# XOR Classification

The "hello world" of neural networks — solving the XOR problem, which a single linear layer cannot do.

```python
import numpy as np
from leanpass import Tensor
from leanpass.nn import MLP, binary_cross_entropy_loss
from leanpass.optim import SGD

# XOR truth
