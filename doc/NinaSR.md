# NinaSR: scalable neural network for Super-Resolution

NinaSR is a neural network to perform super-resolution. It targets a large range of computational budget, from 0.1M to 10M flops/pixel, while aiming for a very short training time.
At the high end, it achieves results similar to RCAN.


## Architecture

I used a simple residual block (two 3x3 convolutions), with a channel attention block. Grouped convolutions tend to be slow on CUDA, so only simple convolutions are used.
After some experiments, the residual block has an expansion ratio of 2x (similar to [WDSR](https://arxiv.org/abs/1808.08718)), and the attention block is local (31x31 average pooling) instead of global.

Deep and narrow networks tend to have better quality, as exemplified by [RCAN](https://arxiv.org/abs/1807.02758). This comes at the cost of a slower training and inference.
I picked parameters that achieve high quality while keeping near-optimal running times on my machine.

The network is initialized following ideas from [NFNet](https://arxiv.org/abs/2102.06171): the layers are scaled to maintain the variance in the residual block, and the second layer of each block is initialized to zero.


## Training

High learning rates are very beneficial for the final accuracy, and tend to shorten the training time.
By default, the training on Div2K takes 300 epochs with Adam, starting with a 1e-3 learning rate. This has to be longer with a slower learning rate for the B2 version.

For higher scales (3x, 4x, 8x), I start by freezing the pretrained network, except the upsampling layer, for one epoch. This allows for a shorter training time.
Finally, the 2x network is retrained, starting from a network pretrained for a higher scaling factor. This improves accuracy even more.



