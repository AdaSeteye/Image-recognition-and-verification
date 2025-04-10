from imports import *
import config

class ResidualModule(torch.nn.Module):
    def __init__(self, input_channels, output_channels, step=1, downsampler=None):
        super(ResidualModule, self).__init__()
        self.convolution1 = torch.nn.Conv2d(input_channels, output_channels, 3, step, 1, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm2d(output_channels)
        self.activation = torch.nn.ReLU()
        self.convolution2 = torch.nn.Conv2d(output_channels, output_channels, 3, 1, padding=1, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm2d(output_channels)
        self.downsampler = downsampler

    def forward(self, input_data):
        residual = input_data
        if self.downsampler is not None:
            residual = self.downsampler(input_data)
        output = self.convolution1(input_data)
        output = self.batch_norm1(output)
        output = self.activation(output)
        output = self.convolution2(output)
        output = self.batch_norm2(output)
        output += residual
        output = self.activation(output)
        return output

class Architecture(torch.nn.Module):
    def __init__(self, num_classes):
        super(Architecture, self).__init__()
        self.initial_convolution = torch.nn.Conv2d(3, 128,7, 4, 3, bias=False)
        self.initial_batch_norm = torch.nn.BatchNorm2d(128)
        self.initial_activation = torch.nn.ReLU()
        
        self.block1 = self._create_block(128, 256, stride=2)
        self.block2 = self._create_block(256, 350, stride=2)
        self.block3 = self._create_block(350, 512, stride=2)
        self.block4 = self._create_block(512, 512, stride=2)
        
        self.average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.flattener = torch.nn.Flatten()
        self.classification_layer = torch.nn.Linear(512, num_classes)
    
    def _create_block(self, input_channels, output_channels, stride):
        downsampler = None
        if stride != 1 or input_channels != output_channels:
            downsampler = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(output_channels),
            )
        return torch.nn.Sequential(
            ResidualModule(input_channels, output_channels, stride, downsampler),
            ResidualModule(output_channels, output_channels)
        )
    
    def forward(self, input_data):
        input_data = self.initial_convolution(input_data)
        input_data = self.initial_batch_norm(input_data)
        input_data = self.initial_activation(input_data)
        
        input_data = self.block1(input_data)
        input_data = self.block2(input_data)
        input_data = self.block3(input_data)
        input_data = self.block4(input_data)
        
        input_data = self.average_pooling(input_data)
        features = self.flattener(input_data)
        output = self.classification_layer(features)
        
        return {"features": features, "output": output}

