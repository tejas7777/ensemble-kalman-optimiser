import torch
from torchviz import make_dot
from model.dnn import MNIST_CNN  # Ensure this path is correct for your project structure
import hiddenlayer as hl

# Instantiate the MNIST_CNN model
model = MNIST_CNN()

# Create a dummy input tensor with the appropriate dimensions
x = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image

# Perform a forward pass to get the output
y = model(x)

# Generate the computation graph
dot = make_dot(y, params=dict(model.named_parameters()))

# Render and save the graph as a PNG file
dot.render("mnist_cnn_model", format="png")