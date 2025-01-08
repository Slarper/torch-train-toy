import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define a list of 10 distinct colors for categories 0 through 9
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

# Create a ListedColormap from the list of colors
cmap = ListedColormap(colors)

# Example data with categories 0 through 9
data = np.random.randint(0, 10, size=100)  # Random integers between 0 and 9

# Generate random x and y coordinates for the scatter plot
x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y, c=data, cmap=cmap)
plt.colorbar(ticks=np.arange(10))
plt.title('Scatter Plot with Discrete Colormap')
plt.savefig('scatter_plot_with_discrete_colormap.png')  # Save the figure as a PNG file

