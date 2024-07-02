from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Reshape, Dot, Activation, Lambda, MaxPool1D, Add, Permute
from tensorflow.keras import backend as K

class NonLocalBlock:
    def __init__(self, intermediate_dim=None, compression=2, strides=1, mode='embedded', add_residual=True):
        """
        Initializes a NonLocalBlock instance.

        Parameters
        ----------
        intermediate_dim: None / int
            The dimension of the intermediate representation. Can be `None` or a positive integer greater than 0. If `None`, computes the intermediate dimension as half of the input channel dimension.
        compression : float, optional
            The factor by which to compress feature dimensions during pooling operations i.e. the pool_size. Defaults to 2, halving the feature dimensions.
        strides : int, optional
            The stride of the pooling operation. Defaults to 1, allowing for overlapping pooling windows.
        mode: str
            Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or `concatenate`.
        add_residual: bool
            Decides if the residual connection should be added or not. Default is True for ResNets, and False for Self Attention.
        """
        self.intermediate_dim = intermediate_dim
        self.compression = compression
        self.strides = strides
        self.mode = mode
        self.add_residual = add_residual

    def __call__(self, ip):
        """
        Applies the Non-Local block to the input tensor.

        Returns:
            Tensor: Output tensor of the Non-Local block with the same shape as the input.

        Parameters
        ----------
        ip: array
            Input tensor.
        """
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(ip)

        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        if self.compression is None:
            self.compression = 1

        # check rank and calculate the input shape
        self.rank = len(input_shape)
        if self.rank not in [3, 4, 5]:
            raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

        elif self.rank == 3:
            batchsize, dims, channels = input_shape

        else:
            if channel_dim == 1:
                batchsize, channels, *dims = input_shape
            else:
                batchsize, *dims, channels = input_shape

        # verify correct intermediate dimension specified
        if self.intermediate_dim is None:
            self.intermediate_dim = channels // 2

            if self.intermediate_dim < 1:
                self.intermediate_dim = 1

        else:
            self.intermediate_dim = int(self.intermediate_dim)

            if self.intermediate_dim < 1:
                raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

        if self.mode == 'gaussian':  # gaussian instantiation
            x1 = Reshape((-1, channels))(ip)  # xi
            x2 = Reshape((-1, channels))(ip)  # xj
            x2 = self.transpose_xj(x2)
            f = Dot(axes=[self.rank-1, self.rank-1])([x1, x2])
            f = Activation('softmax')(f)
        elif self.mode == 'dot':  # dot instantiation
            # theta path
            theta = self._convND(ip, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)
            # phi path
            phi = self._convND(ip, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)
            phi = self.transpose_xj(phi)
            f = Dot(axes=[self.rank-1, self.rank-1])([theta, phi])
            # scale the values to make it size invariant
            f = Lambda(lambda z: (1. / float(K.int_shape(f)[-1])) * z)(f)  
        elif self.mode == 'concatenate':  # concatenation instantiation
            raise NotImplementedError('Concatenate model has not been implemented yet')
        else:  # Embedded Gaussian instantiation
            # theta path
            theta = self._convND(ip, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)
            # phi path
            phi = self._convND(ip, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)
            phi = self.subsampling_trick(phi)
            phi = self.transpose_xj(phi)
            f = Dot(axes=[self.rank-1, self.rank-2])([theta, phi])
            f = Activation('softmax')(f)

        # g path
        g = self._convND(ip, self.intermediate_dim)
        g = Reshape((-1, self.intermediate_dim))(g)
        if self.mode == 'embedded':
            g = self.subsampling_trick(g)

        # compute output path
        y = Dot(axes=[self.rank-1, self.rank-2])([f, g])

        # reshape to input tensor format
        if self.rank == 3:
            y = Reshape((dims, self.intermediate_dim))(y)
        else:
            if channel_dim == -1:
                y = Reshape((*dims, self.intermediate_dim))(y)
            else:
                y = Reshape((self.intermediate_dim, *dims))(y)

        # project filters
        y = self._convND(y, channels)

        # residual connection
        if self.add_residual:
            y = Add()([ip, y])

        return y
    
    def transpose_xj(self, xj): return Permute((self.rank-1, self.rank-2))(xj)

    def subsampling_trick(self, x):
        # Subsampling trick for a more sparse computation (Wang et. al 2018)
        print('compression', self.compression)
        if self.compression > 1:
            x = MaxPool1D(pool_size=self.compression, strides=self.strides)(x)  # Apply compression as max pooling
        return x

    def _convND(self, ip, channels):
        """
        Applies a convolution operation based on the rank of the input tensor.

        Returns:
            Tensor: Output of the convolution operation.

        Parameters
        ----------
        ip: array
            Input tensor.
        channels: int 
            Number of output channels for the convolution.
        """
            
        assert self.rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if self.rank == 3:
            x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        elif self.rank == 4:
            x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        else:
            x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        return x
