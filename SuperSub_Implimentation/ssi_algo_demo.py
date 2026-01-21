"""
This file is the demo / proof of concept for the Super/Sub implimentation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib import animation


def shape_traverse(shape):
    """
    The main algorithm, inputs are the nodes and their connections
    Output is path of traversal for each instance of each node
    """
    # Step 1, generating a array of how far away each node is from the end, assuming first and last entry is entry and exit node respectively 
    start_node = 0 
    end_node = len(shape)-1
    distance = generate_distance(end_node, shape)

    # Step 2, generating all legal paths an instance could take to reach the end with recursion
    legal_paths = generate_paths(shape, start_node, end_node)
    print(legal_paths)
    return legal_paths

def generate_paths(graph, start, end):
    """
    Implimentation of simple backtrack DFS Algo
    Gets all simple paths of all lengths with  no repeating nodes
    """

    all_paths = []

    def backtrack(current_node, visited, current_path):
        if current_node == end:
            all_paths.append(current_path.copy())
            return
        
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                current_path.append(neighbor)

                backtrack(neighbor, visited, current_path)

                current_path.pop()
                visited.remove(neighbor)

    backtrack(start, {start}, [start])
    return all_paths

def generate_distance(end_node, shape):
    """
    Algorithm to determine how far each node is from the exit node 
    Exit node is assumed to be the last node in the shape data structure
    """
    distance = np.full((end_node+1), end_node+1)
    toVisit = np.array([], dtype=int)
    currentNode = end_node
    currentVisitIndex = 0

    distance[end_node] = 0 # Setting initial value of exit node
    while end_node+1 in distance:
        for connectionNode in shape[currentNode]:
            if distance[connectionNode] == end_node+1: 
                # adding distance to list
                distance[connectionNode] = distance[currentNode]+1
                # adding node to list to check next
                toVisit = np.append(toVisit, connectionNode)
        currentNode = toVisit[currentVisitIndex]
        currentVisitIndex += 1

    return distance 


def animate_paths_3d(connection_dict, paths, filename="paths.gif", interval=800):
    """
    Animates paths one-by-one and saves as a GIF
    """

    coords = {
        0: (0,1,0), 1: (1,1,0), 2: (0,0,0), 3: (1,0,0),
        4: (0,1,1), 5: (1,1,1), 6: (0,0,1), 7: (1,0,1)
    }

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")

    def draw_base():
        ax.clear()

        for i, (x,y,z) in coords.items():
            ax.scatter(x, y, z, color="black")
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

                ax.plot(x, y, z, color="lightgrey", linewidth=1)

        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Animating Paths")

    cmap = plt.cm.get_cmap("tab10", len(paths))

    def update(frame):
        draw_base()

        for i in range(frame + 1):
            color = cmap(i)
            path = paths[i]

            for a, b in zip(path[:-1], path[1:]):
                x = [coords[a][0], coords[b][0]]
                y = [coords[a][1], coords[b][1]]
                z = [coords[a][2], coords[b][2]]

                ax.plot(x, y, z, color=color, linewidth=3)

        ax.set_title(f"Path {frame+1}/{len(paths)}")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(paths),
        interval=interval,
        repeat=False
    )

    save_path = os.path.join(os.getcwd(), filename)
    anim.save(save_path, writer="pillow", dpi=150)
    plt.close()

    print(f"Animation saved to: {save_path}")


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
    paths = shape_traverse(shape)
    animate_paths_3d(shape, paths)

main()