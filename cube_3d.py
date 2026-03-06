import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from hand_tracker import HandTracker  # Assume you have a hand tracking class

def draw_cube(ax, size=1, position=(0, 0, 0), color='b'):
    r = size / 2  # Half size
    # Define vertices of the cube
    vertices = np.array([[r, r, r], [r, r, -r], [r, -r, r], [r, -r, -r],
                         [-r, r, r], [-r, r, -r], [-r, -r, r], [-r, -r, -r]])
    vertices += np.array(position)  # Translate the cube to the position

    # Define the six faces of the cube
    faces = [[vertices[j] for j in [0, 1, 3, 2]],
             [vertices[j] for j in [4, 5, 7, 6]],
             [vertices[j] for j in [0, 2, 6, 4]],
             [vertices[j] for j in [1, 3, 7, 5]],
             [vertices[j] for j in [0, 4, 5, 1]],
             [vertices[j] for j in [2, 3, 7, 6]]]

    # Draw the cube
    for face in faces:
        ax.add_collection3d(plt.Polygon(face, color=color, alpha=0.5))

def main():
    tracker = HandTracker()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    while True:
        hand_position = tracker.get_hand_position()  # Assume this gets the hand position
        ax.cla()  # Clear the axes
        draw_cube(ax, position=hand_position)
        
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_zlim([-5, 5])
        plt.draw()
        plt.pause(0.1)

if __name__ == "__main__":
    main()