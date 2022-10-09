from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Dropout

__all__ = ['cyclone_classifier']

#creates a single, non-residual, convolutional block
#   filters - the number of filters/channels in the block
#   kernel_size - the "area" of the convolution kernel
#   alpha - negative slope coefficient in leaky relu layer
def conv_block(filters: int, kernel_size: int, alpha=0.1, name=None) -> Sequential:
    return Sequential(
        [
            Conv2D(filters, kernel_size, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=alpha)
        ],
        name=name
    )

#creates a single residual block, which includes *two* convolutional layers, norms, and activations
#   filters - the number of filters/channels in the block
#   kernel_size - the "area" of the convolution kernel
#   alpha - negative slope coefficient in leaky relu layer
class ResidualBlock(Layer):

    def __init__(self, filters: int, kernel_size: int, alpha=0.1):
        super(ResidualBlock, self).__init__()
        self.block = Sequential([
            Conv2D(filters, kernel_size, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=alpha),
            Conv2D(filters, kernel_size, padding='same', use_bias=False),
            BatchNormalization()
        ])
        self.activation = LeakyReLU(alpha=alpha)

    def build(self, shape):
        self.block.build(shape)
        self.activation.build(shape)

    def call(self, inputs):
        outputs = self.block(inputs)
        outputs += inputs
        return self.activation(outputs)

#forms a whole residual stage, stacking any number of residual blocks
#   filters - the number of filters/channels in each block
#   kernel_size - the "area" of the convolution kernel in each block
#   blocks - number of individual residual blocks to stack
#   alpha - negative slope coefficient in leaky relu layer
def residual_stage(filters: int, kernel_size: int, blocks: int, alpha=0.1, name=None) -> Sequential:
    assert blocks >= 1
    stage = []
    for _ in range(blocks):
        stage.append( ResidualBlock(filters, kernel_size, alpha) )
    return Sequential(stage, name=name)

#stacks together convolutional and residual blocks with 5 pooling layers and sigmoid output
#   channels - list of channels/depth of all 6 stages
#   kernels - list of kernel sizes for all 6 stages
#   blocks - list of number of residual blocks for all 6 stages
#   dropout - fractional dropout after each stage
#   alpha - negative slope coefficient in leaky relu layer
def cyclone_classifier(channels: list, kernels: list, blocks: list, dropout: float, alpha=0.1) -> Sequential:

    for params in (channels, kernels, blocks):
        assert type(params) is list and len(params) == 6

    model = []
    for i in range(5):
        model += [
            conv_block(
                channels[i],
                kernels[i],
                alpha,
                f'conv_block_{i}_c{channels[i]}_k{kernels[i]}'
            ),
            residual_stage(
                channels[i],
                kernels[i],
                blocks[i],
                alpha,
                f'residual_stage_{i}_c{channels[i]}_k{kernels[i]}_b{blocks[i]}'
            ),
            MaxPool2D(),
            Dropout(dropout)
        ]

    model += [
        conv_block(
            channels[-1],
            kernels[-1],
            alpha,
            f'conv_block_6_c{channels[i]}_k{kernels[i]}'
        ),
        residual_stage(
            channels[-1],
            kernels[-1],
            blocks[-1],
            alpha,
            f'residual_stage_6_c{channels[i]}_k{kernels[i]}_b{blocks[i]}'
        ),
        Conv2D(1, 3, padding='same', activation='sigmoid')
    ]

    return Sequential(model, name='cyclone_classifier')