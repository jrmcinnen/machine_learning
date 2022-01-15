# DATA.ML.100
# Jere Mäkinen
# jere.makinen@tuni.fi

"""
Linear regression

Program creates a grid. User can then click points into the grid using left click. Once the user is finished with
inserting the points, they can fit a line into set of points using right-click.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Cursor

# Linear Solver from exercise 1b
def mylinfit(x, y):
    x, y = np.array(x), np.array(y)
    n = len(x)
    b = (sum(y)-sum(x)*sum(x*y)/sum(x**2))/(n*(1-sum(x)*sum(x)/(n*sum(x**2))))
    a = (sum(x*y)-b*sum(x))/sum(x**2)
    return a, b


# Initialize the figure
axis_size = 10
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-axis_size, axis_size])
ax.set_ylim([-axis_size, axis_size])
ax.set_title('Left click to enter points and right click to fit a line')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

# Add a cursor to make it easier to add points
cursor = Cursor(ax, horizOn=True, vertOn=True, color='green')

# Initialize containers for mouse click coordinates
x = []
y = []


# Click handler
def on_click(event):
    if event.button is MouseButton.LEFT:
        # Save the data from the mouse click
        ix, iy = float(event.xdata), float(event.ydata)
        x.append(ix)
        y.append(iy)
        # Plot the point
        plt.scatter(ix,iy,c='b')
        fig.canvas.draw()
    elif event.button is MouseButton.RIGHT:
        # Compute the cooefficients a and b
        a,b = mylinfit(x, y)
        # Plot the fitted linear model
        xp = np.linspace(-axis_size, axis_size, 1000)
        plt.plot(xp,a*xp+b,'r-', label=f'{a}*x + {b}')
        plt.legend()
        fig.canvas.draw()
        # Stop tracking mouse clicks
        plt.disconnect(binding_id)

binding_id = plt.connect('button_press_event', on_click)

plt.show()





