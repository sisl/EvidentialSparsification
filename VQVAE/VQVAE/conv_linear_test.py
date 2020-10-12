import torch

inputs = torch.tensor([[[[1., 2.],[3., 4.]]]])
print("linear inputs", inputs.shape)

fc = torch.nn.Linear(4,2)
weights = torch.tensor([[1.1, 1.2, 1.3, 1.4],
						[1.5, 1.6, 1.7, 1.8]])
bias = torch.tensor([1.9, 2.0])
print("linear weights", weights.shape)
fc.weight.data = weights
fc.bias.data = bias
output = torch.relu(fc(inputs.view(-1,4)))
print("linear outputs", output, output.shape)

conv = torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1,1))
print("conv inputs", inputs.view(1,4,1,1).shape)
conv.weight.data = weights.view(2,4,1,1)
print("conv weights", weights.view(2,4,1,1).shape)
conv.bias.data = bias
output_conv = torch.relu(conv(inputs.view(1,4,1,1)))
print("conv outputs", output_conv, output_conv.shape)