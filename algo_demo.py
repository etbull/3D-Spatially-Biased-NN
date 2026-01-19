"""
This file is the demo / proof of concept for the Super/Sub implimentation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def shape_traverse(shape):
    """
    The main algorithm, inputs are the nodes and their connections
    Output is path of traversal for each instance of each node
    """
    # Step 1, generating a dict of how far away each node is from the end, assuming first and last entry is first and last node
    start_node = 0 
    end_node = len(shape)-1
    distance = generate_distance(start_node, end_node, shape)

def generate_distance(start_node, end_node, shape):
    distance = 


def plot_3d_graph(connection_dict, filename="cube_graph.png"):
    """
    Visualises the connections of the cubes
    """

    coords = {
        0: (0,1,0), 1: (1,1,0), 2: (0,0,0), 3: (1,0,0),
        4: (0,1,1), 5: (1,1,1), 6: (0,0,1), 7: (1,0,1)
    }

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    for i, (x,y,z) in coords.items():
        ax.scatter(x, y, z)
        ax.text(x, y, z, f"{i}", size=10)

    drawn = set()
    for src, targets in connection_dict.items():
        for tgt in targets:
            edge = tuple(sorted((src, tgt)))
            if edge in drawn:
                continue
            drawn.add(edge)

            x = [coords[src][0], coords[tgt][0]]
            y = [coords[src][1], coords[tgt][1]]
            z = [coords[src][2], coords[tgt][2]]

            ax.plot(x, y, z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Node Connectivity")

    ax.set_box_aspect([1,1,1])
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), filename)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Graph saved to: {save_path}")

def main():
    """
    Demonstration implimentation main function
    """
    shape = {
        0:[1,2,4],
        1:[0,3,5],
        2:[0,3,6],
        3:[1,2,7],
        4:[0,5,6],
        5:[1,4,7],
        6:[2,4,7],
        7:[3,5,6]
    }
    plot_3d_graph(shape)
    shape_traverse(shape)

main()