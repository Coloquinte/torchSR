# NinaSR: scalable neural network for Super-Resolution

NinaSR is a neural network to perform super-resolution. It targets a large range of computational budget, from 0.1M to 10M flops/pixel, while aiming for a very short training time.


## Architecture

I used a simple residual block (two 3x3 convolutions), with a channel attention block.
After some experiments, the residual block has an expansion ratio of 2x, and the attention block is local (15x15 average pooling) instead of global.

Deep and narrow networks tend to have better quality, as exemplified by RCAN. This comes at the cost of a slower training and inference.
I picked parameters that achieve high quality while keeping near-optimal running times on my machine.

The network is initialized following ideas from NFNet: the layers are scaled to maintain the variance in the residual block, and the second layer of each block is initialized to zero.


## Training

High learning rates are very beneficial for the final accuracy, and tend to shorten the training time.
By default, the training on Div2K takes 300 epochs with Adam, starting with a 1e-3 learning rate.

For higher scales (3x, 4x, 8x), I start by freezing the network, except the upsampling layer, for one epoch. This allows for an extremely short training time.
