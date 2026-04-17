# Abel Modular Layers

A clean, dimension-agnostic PyTorch block system for 1D and 2D models.

## Class Hierarchy

- `BaseBlock`
  - `ConvNormActBlock`
  - `ConvNetBlock`
    - `ConvNetUpsample`
    - `ConvNetUpsampleTrain`
    - `ConvNetDownsample`
    - `ConvNetDownsampleTrain`
  - `ResidualBlock`
    - `ResNetUpsample`
    - `ResNetUpsampleTrain`
    - `ResNetDownsample`
    - `ResNetDownsampleTrain`

## Composition Pieces

- `LayerFactory`: selects `Conv1d/Conv2d`, `ConvTranspose1d/ConvTranspose2d`, pooling layers by `dim`.
- Sampling modules (fully decoupled):
  - `StaticUpsample`, `StaticDownsample`
  - `TrainableUpsample`, `TrainableDownsample`
- Activations:
  - `build_activation` with `leaky_relu`, `relu`, `gelu`, `silu`, `snake`, `identity`

## Factory API

`BlockFactory(dim=1 or 2)` creates blocks without duplicating call-sites:

- `convnet(...)`
- `convnet_upsample(..., trainable=False|True)`
- `convnet_downsample(..., trainable=False|True)`
- `resnet(...)`
- `resnet_upsample(..., trainable=False|True)`
- `resnet_downsample(..., trainable=False|True)`

## Example

```python
import torch
from Modules.abel import BlockFactory

factory = BlockFactory(dim=1)
block = factory.resnet_downsample(
    in_channels=128,
    out_channels=256,
    trainable=True,
    kernel_size=3,
    bottleneck=True,
    expansion=4,
)

x = torch.randn(4, 128, 320)
y = block(x)
print(y.shape)  # [4, 256, 160]
```
