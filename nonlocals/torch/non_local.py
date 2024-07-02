import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, intermediate_dim=None, compression=2, strides=1, mode='embedded', add_residual=True):
        super(NonLocalBlock, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.compression = compression if compression is not None else 1
        self.strides = strides
        self.mode = mode
        self.add_residual = add_residual
        self.conv_layers = nn.ModuleDict()

    def build(self, input_shape):
        channels = input_shape[1]
        self.intermediate_dim = channels // 2 if self.intermediate_dim is None else int(self.intermediate_dim)
        if self.intermediate_dim < 1:
            self.intermediate_dim = 1

        # Instantiate convolution layers here to be used in the forward method
        self.rank = len(input_shape)
        self.conv_layers['theta'] = self._create_conv_layer(channels, self.intermediate_dim)
        self.conv_layers['phi'] = self._create_conv_layer(channels, self.intermediate_dim)
        self.conv_layers['g'] = self._create_conv_layer(channels, self.intermediate_dim)
        self.conv_layers['final'] = self._create_conv_layer(self.intermediate_dim, channels)

    def forward(self, inputs):
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
        channels = inputs.shape[1]

        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`, or `concatenate`')
        return channels

    def _instantiate_f(self, channels, theta, phi, inputs):
        if self.mode == 'gaussian':
            xi = inputs.view(inputs.size(0), channels, -1)
            xj = inputs.view(inputs.size(0), channels, -1).permute(0, 2, 1)
            f = torch.einsum('bic,bcj->bij', xi, xj)
            f = F.softmax(f, dim=-1)
        elif self.mode == 'dot':
            theta = theta.view(theta.size(0), self.intermediate_dim, -1)
            phi = phi.view(phi.size(0), self.intermediate_dim, -1).permute(0, 2, 1)
            f = torch.einsum('bic,bcj->bij', theta, phi)
            f = (1. / float(f.size(-1))) * f
            f = F.softmax(f, dim=-1)
        elif self.mode == 'embedded':
            theta = theta.view(theta.size(0), self.intermediate_dim, -1)
            phi = phi.view(phi.size(0), self.intermediate_dim, -1).permute(0, 2, 1)
            phi = self.subsampling_trick(phi)
            
            print(f"Shape of theta: {theta.shape}")
            print(f"Shape of phi: {phi.shape}")
            f = torch.einsum('bic,bcj->bij', theta, phi)
            print(f"Shape of f: {f.shape}")
            
            f = F.softmax(f, dim=-1)
        else:
            raise NotImplementedError('Concatenate mode has not been implemented yet')

        return f

    def _handle_g(self, g):
        g = g.view(g.size(0), self.intermediate_dim, -1)
        if self.mode == 'embedded':
            g = self.subsampling_trick(g.permute(0, 2, 1)).permute(0, 2, 1)
        print(f"Shape of g: {g.shape}")

        return g

    def subsampling_trick(self, x):
        if self.compression > 1:
            x = F.max_pool1d(x, kernel_size=self.compression, stride=self.strides)
        return x

    def _non_local_operation_neural(self, f, g, inputs):
        y = torch.matmul(f, g)
        y = y.view(inputs.size(0), self.intermediate_dim, *inputs.shape[2:])
        return y

    def _non_local_block(self, y, inputs):
        y = self.conv_layers['final'](y)
        if self.add_residual:
            y = y + inputs
        return y

    def _create_conv_layer(self, in_channels, out_channels):
        if self.rank == 3:
            return nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        elif self.rank == 4:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        elif self.rank == 5:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
