#!/usr/bin/env python 

# settings for plotting
WHITE_BACKGROUND = True # sets white background on plotting (only work for total spike count)
SUM_POLARITIES = True # negative and positive polarities are summed together



import numpy as np
from matplotlib import pyplot as plt
import rospy
from std_msgs.msg import String
from matplotlib import animation
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from std_msgs.msg import String, Float64MultiArray

fig, ax = plt.subplots()
ax = Axes3D(fig)
#ax = fig.gca(projection='polar')
fig.canvas.set_window_title('BioTac Sensor')
fig.patch.set_facecolor('white')
origin = 0, 0, 0 # origin point
Q = ax.quiver([0], [0], [0], [1], [1], [1], color='b')
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_zlim([-10,10])
ax.set_xlabel('X')
ax.set_xlabel('Y')
ax.set_xlabel('Z')

X = 1.0
Y = 1.0
Z = 1.0


def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

# initialization function: plot the background of each frame
def init():
    global Q,X,Y,Z
    segments = quiver_data_to_segments(0,0,0, X,Y,Z)
    Q.set_segments(segments)
    return Q

# animation function.  This is called sequentially
def animate(i):
    global Q,X,Y,Z
    segments = quiver_data_to_segments(0,0,0, X,Y,Z)
    Q.set_segments(segments)
    return Q

def assign_vector(msg):
    global X,Y,Z
    X = msg.data[5]
    Y = msg.data[6]
    Z = msg.data[7]
    print(X,Y,Z, np.sqrt(X**2+Y**2+Z**2))

if __name__ == '__main__':

    rospy.init_node("plotter")
    rospy.Subscriber("touched", Float64MultiArray, assign_vector)

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=5, blit=False)
    plt.show()
    rospy.spin()