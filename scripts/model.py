from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Dropout

__all__ = ['cyclone_classifier']

def conv_block(filters: int, kernel_size: int, alpha=0.1) -> Sequential:
    return Sequential([
        Conv2D(filters, kernel_size, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=alpha)
    ])

class ResidualBlock(Layer):

    def __init__(self, filters: int, kernel_size: int):
        super(ResidualBlock, self).__init__()
        self.block = Sequential([
            Conv2D(filters, kernel_size, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Conv2D(filters, kernel_size, padding='same', use_bias=False),
            BatchNormalization()
        ])
        self.activation = LeakyReLU(alpha=0.1)

    def build(self, shape):
        self.block.build(shape)
        self.activation.build(shape)

    def call(self, inputs):
        outputs = self.block(inputs)
        outputs += inputs
        return self.activation(outputs)

def residual_stage(filters: int, kernel_size: int, blocks: int) -> Sequential:
    assert blocks >= 1
    stage = []
    for _ in range(blocks):
        stage.append( ResidualBlock(filters, kernel_size) )
    return Sequential(stage)

def cyclone_classifier(
        channels: list,
        kernels: list,
        blocks: list,
        dropout: float) -> Sequential:

    for params in (channels, kernels, blocks):
        assert type(params) is list and len(params) == 6
    model = []
    for i in range(5):
        model += [
            conv_block(channels[i], kernels[i]),
            residual_stage(channels[i], kernels[i], blocks[i]),
            MaxPool2D(),
            Dropout(dropout)
        ]
    model += [
        conv_block(channels[-1], kernels[-1]),
        residual_stage(channels[-1], kernels[-1], blocks[-1]),
        Conv2D(1, 3, padding='same', activation='sigmoid')
    ]
    return Sequential(model)