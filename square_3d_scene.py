from manimlib import *
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # A simple network with one hidden layer
        self.fc1 = nn.Linear(1, 3)  # 1 input, 64 neurons in the hidden layer
        self.fc11 = nn.Linear(3, 3)
        self.fc12 = nn.Linear(3, 3)
        self.fc13 = nn.Linear(3, 3)
        self.fc14 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)  # 64 neurons to 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = self.fc2(x)
        return x


model = SimpleNN()
model_weights = load_file("models/square_3d.safetensors")
model.load_state_dict(model_weights)

# Plot the results
model.eval()


def dots_array_transform(scene, dots1, dots2, run_time=5):
    scene.play(
        *(Transform(dot1, dot2) for dot1, dot2 in zip(dots1, dots2)), run_time=run_time
    )


def to_dots(from_, axes, padding=0):

    colors = color_gradient([BLUE, RED], len(from_))
    if padding == 0:

        dots2 = Group()
        for xyz, color in zip(from_, colors):
            dot = TrueDot(center=axes.c2p(*xyz), color=color)
            dots2.add(dot)
        return dots2

    if padding == 2:
        dots = Group()
        for x, color in zip(from_, colors):
            dot = TrueDot(center=axes.c2p(*(x, 0, 0)), color=color)
            dots.add(dot)
        return dots

'''
all hidden layers are 3-by-3-dimensional fully connection layers
'''
def transform_between_hidden_layers(
        xyz1,
        functionals,
        axes,
        scene
):

    for functional in functionals:
        dots2 = to_dots(xyz1.detach().numpy(), axes)
        xyz1 = functional(xyz1) # Update xyz1

        dots3 = to_dots(xyz1.detach().numpy(), axes)

        dots_array_transform(scene, dots2, dots3, run_time=2)

        scene.wait()
        scene.remove(dots3)
        scene.remove(dots2)

    return xyz1


class InteractiveDevelopment(Scene):
    def construct(self):
        self.wait()

        # This opens an iPython terminal where you can keep writing
        # lines as if they were part of this construct method.
        # In particular, 'square', 'circle' and 'self' will all be
        # part of the local namespace in that terminal.
        # self.embed()

        axes = ThreeDAxes(
            # x-axis ranges from -1 to 10, with a default step size of 1
            x_range=(-20, 20, 2),
            # y-axis ranges from -2 to 2 with a step size of 0.5
            y_range=(-20, 20, 2),
            z_range=(-20, 20, 2),
            # The axes will be stretched so as to match the specified
            # height and width
            height=16,
            width=16,
            depth=16,
        )
        axes.set_width(FRAME_WIDTH)
        axes.center()

        self.frame.reorient(43 + 90, 76, 1, OUT, 8)
        self.add(axes)

        self.wait()

        x_samples = np.linspace(-10, 10, 107)
        x_samples = torch.tensor(x_samples, dtype=torch.float32).view(-1, 1)

        x_samples.numpy().shape

        dots = to_dots(x_samples.numpy(), axes, padding=2)

        self.add(dots)

        xyz1 = model.fc1(x_samples)

        dots2 = to_dots(xyz1.detach().numpy(), axes)

        self.add(dots2)
        self.remove(dots2)

        dots_array_transform(
            self,
            dots,
            dots2,
            run_time = 2
        )

        self.wait()
        self.remove(dots)
        self.remove(dots2)

        xyz = transform_between_hidden_layers(
            xyz1,
            [torch.relu, model.fc11, torch.relu,
             model.fc12, torch.relu, model.fc13, torch.relu,
             model.fc14, torch.relu
             ],
            axes,
            self
        )
        dots1 = to_dots(xyz.detach().numpy(), axes)

        xyz = model.fc2(xyz)
        dots2 = to_dots(xyz.detach().numpy(), axes, padding = 2)

        dots_array_transform(
            self,
            dots1,
            dots2,
            run_time =2
        )

        self.embed()

        self.clear()
