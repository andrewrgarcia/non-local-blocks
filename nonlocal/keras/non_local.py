from tensorflow.keras.layers import Layer, Conv1D, Conv2D, Conv3D, Reshape, Dot, Activation, Lambda, MaxPool1D, Add, Permute
from tensorflow.keras import backend as K

class NonLocalBlock(Layer):
    def __init__(self, intermediate_dim=None, compression=2, strides=1, mode='embedded', add_residual=True, **kwargs):
        """
        Initializes a NonLocalBlock with configurable parameters and operational mode.

        Parameters
        ----------
        intermediate_dim : int, optional
            The dimension of the intermediate representation in the convolution layers. If None, it defaults to half of the channels in the input shape.
        compression : float, optional
            The factor by which to compress feature dimensions during pooling operations i.e. the pool_size. Defaults to 2, halving the feature dimensions.
        strides : int, optional
            The stride of the pooling operation. Defaults to 1, allowing for overlapping pooling windows.
        mode : str, optional
            Operational mode of the block. Supported modes are 'gaussian', 'dot', 'embedded', and 'concatenate'. Defaults to 'embedded'.
        add_residual : bool, optional
            Whether to include a residual connection that adds the input to the output of the block. Defaults to True.
        kwargs : dict
            Additional keyword arguments inherited from tf.keras.layers.Layer.
        """
        super(NonLocalBlock, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.compression = compression if compression is not None else 1
        self.strides = strides
        self.mode = mode
        self.add_residual = add_residual
        self.conv_layers = {}

    def build(self, input_shape):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = input_shape[channel_dim]
        self.intermediate_dim = channels // 2 if self.intermediate_dim is None else int(self.intermediate_dim)
        if self.intermediate_dim < 1:
            self.intermediate_dim = 1

        # Instantiate convolution layers here to be used in the call method
        self.rank = len(input_shape)
        self.conv_layers['theta'] = self._create_conv_layer(self.intermediate_dim)
        self.conv_layers['phi'] = self._create_conv_layer(self.intermediate_dim)
        self.conv_layers['g'] = self._create_conv_layer(self.intermediate_dim)
        self.conv_layers['final'] = self._create_conv_layer(channels)

    def call(self, inputs):
        # Use the convolution layers created in build
        theta = self.conv_layers['theta'](inputs)
        phi = self.conv_layers['phi'](inputs)
        g = self.conv_layers['g'](inputs)

        channels = self._initialize_dimensions(inputs)

        f = self._instantiate_f(channels, theta, phi, inputs)
        g = self._handle_g(g)
        y = self._non_local_operation_neural(f, g, inputs)
        z = self._non_local_block(y, inputs)

        return z

    def _initialize_dimensions(self, inputs):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = K.int_shape(inputs)[channel_dim]

        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`, or `concatenate`')
        return channels

    def _instantiate_f(self, channels, theta, phi, inputs):
        if self.mode == 'gaussian':
            xi = Reshape((-1, channels))(inputs)
            xj = Reshape((-1, channels))(inputs)
            xj = self.transpose_xj(xj)
            f = Dot(axes=[self.rank-1, self.rank-1])([xi, xj])
            f = Activation('softmax')(f)
        elif self.mode == 'dot':
            theta = Reshape((-1, self.intermediate_dim))(theta)
            phi = Reshape((-1, self.intermediate_dim))(phi)
            phi = self.transpose_xj(phi)
            f = Dot(axes=[self.rank-1, self.rank-1])([theta, phi])
            f = Lambda(lambda z: (1. / float(K.int_shape(f)[-1])) * z)(f)  # reintroduced scaling
            f = Activation('softmax')(f)
        elif self.mode == 'embedded':
            # Embedded Gaussian instantiation
            theta = Reshape((-1, self.intermediate_dim))(theta)
            phi = Reshape((-1, self.intermediate_dim))(phi)
            phi = self.subsampling_trick(phi)
            phi = self.transpose_xj(phi)
            print(f"shape theta {K.int_shape(theta)} shape phi {K.int_shape(phi)}")
            # Ensure phi has correct shape for dot product to produce THW x THW
            f = Dot(axes=[self.rank-1, self.rank-2])([theta, phi])
            print("Shape before softmax:", K.int_shape(f))  # Debug statement
            f = Activation('softmax')(f)
        else:
            # If concatenate mode or any other mode, handle accordingly
            raise NotImplementedError('Concatenate mode has not been implemented yet')

        return f
    
    def transpose_xj(self, xj): return Permute((self.rank-1, self.rank-2))(xj)

    def _handle_g(self, g):
        # g has already been instantiated with a simple linear embedding at the beginning of call 
        g = Reshape((-1, self.intermediate_dim))(g)
        if self.mode == 'embedded':
            g = self.subsampling_trick(g)
        return g
        
    def subsampling_trick(self, x):
        # Subsampling trick for a more sparse computation (Wang et. al 2018)
        print('compression', self.compression)
        if self.compression > 1:
            x = MaxPool1D(pool_size=self.compression, strides=self.strides)(x)  # Apply compression as max pooling
        return x
    
    def _non_local_operation_neural(self, f, g, inputs):
        # Final output path 
        y = Dot(axes=[self.rank-1, self.rank-2])([f, g])
        y = Reshape(K.int_shape(inputs)[1:-1] + (self.intermediate_dim,))(y)
        return y 

    def _non_local_block(self, y, inputs):
        # Final combination and residual connection
        y = self.conv_layers['final'](y)
        if self.add_residual:
            y = Add()([inputs, y])
        return y 

    def _create_conv_layer(self, channels):
        if self.rank == 3:
            return Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')
        elif self.rank == 4:
            return Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')
        elif self.rank == 5:
            return Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(NonLocalBlock, self).get_config()
        config.update({
            "intermediate_dim": self.intermediate_dim,
            "compression": self.compression,
            "strides": self.strides,
            "mode": self.mode,
            "add_residual": self.add_residual
        })
        return config
