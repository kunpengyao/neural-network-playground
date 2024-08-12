import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

def plot_3d_xor():
    fig = plt.figure(figsize=(14, 7))
    
    # 3D plot
    ax = fig.add_subplot(121, projection='3d')
    
    # Create a grid of points covering the input space
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Calculate the XOR for each point in the grid
    zz = np.logical_xor(xx > 0.5, yy > 0.5).astype(int)
    
    # Plot the surface
    surface = ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='coolwarm')

    # Plot the XOR points
    data_points = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    for point in data_points:
        ax.scatter(point[0], point[1], point[2], color='blue' if point[2] == 0 else 'red', s=50, edgecolors='k')

    # Add labels and title
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('XOR Output')
    ax.set_title('3D Visualization of XOR Function')

    # 2D plot (top-down view)
    ax2 = fig.add_subplot(122)
    
    # Plot the XOR surface in 2D
    contour = ax2.contourf(xx, yy, zz, alpha=0.5, cmap='coolwarm')

    # Plot the XOR points in 2D
    for point in data_points:
        ax2.scatter(point[0], point[1], color='blue' if point[2] == 0 else 'red', s=50, edgecolors='k')

    # Add labels and title
    ax2.set_xlabel('Input 1')
    ax2.set_ylabel('Input 2')
    ax2.set_title('2D Top-Down View of XOR Function')
    
    # Add a color bar for the legend
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('XOR Output')

    plt.show()


def visualize_nn(nn):
    G = nx.DiGraph()
    
    # Define positions for nodes in each layer
    positions = {}
    layer_indices = [0] + list(np.cumsum(nn.layer_sizes))
    
    # Add nodes and positions for the input layer
    input_layer_size = nn.layer_sizes[0]
    for i in range(input_layer_size):
        node = f'Input {i+1}'
        positions[node] = (0, input_layer_size - i)
        G.add_node(node)

    # Add nodes and positions for hidden and output layers
    for layer in range(1, len(nn.layer_sizes)):
        layer_size = nn.layer_sizes[layer]
        for i in range(layer_size):
            # node = f'Layer {layer} Neuron {i+1}'
            node = f'L{layer} N{i+1}'
            positions[node] = (layer, layer_size - i)
            G.add_node(node)

    # Add edges with weights
    for layer in range(len(nn.weights)):
        weight_matrix = nn.weights[layer]
        for i in range(weight_matrix.shape[0]):
            for j in range(weight_matrix.shape[1]):
                from_node = f'Input {i+1}' if layer == 0 else f'L{layer} N{i+1}'
                to_node = f'L{layer+1} N{j+1}'
                G.add_edge(from_node, to_node, weight=weight_matrix[i, j])

    # Draw the nodes
    nx.draw_networkx_nodes(G, positions, node_size=2000, node_color='lightblue')
    
    # Draw the edges
    nx.draw_networkx_edges(G, positions, arrowstyle='-|>', arrowsize=20, edge_color='gray')

    # Draw the labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, positions, labels, font_size=10)
    
    # Draw edge labels with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    formatted_edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, positions, edge_labels=formatted_edge_labels, font_size=8)
    
    # Display the plot
    plt.axis('off')
    plt.show()


def evaluate_performance(nn, N):
    # Generate N random testing samples with float values
    test_input_vectors = np.random.rand(N, 2)  # Random floats between 0 and 2
    test_targets = np.array([(1 if x[0] > 0.5 else 0) ^ (1 if x[1] > 0.5 else 0) for x in test_input_vectors])
    
    # Predict using the trained neural network
    correct_predictions = 0
    for input_vector, target in zip(test_input_vectors, test_targets):
        output = nn.predict(input_vector)
        predicted = 1 if output >= 0.5 else 0  # Threshold at 0.5
        if predicted == target:
            correct_predictions += 1
    
    # Calculate success rate
    success_rate = correct_predictions / N
    print(f"Success Rate: {success_rate * 100:.2f}%")
    return success_rate, test_input_vectors, test_targets


# Visualization function for training and testing data
def visualize_data(train_input_vectors, test_input_vectors, train_targets, test_targets, nn):
    plt.figure(figsize=(8, 6))

    # Create a grid of points covering the input space
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict the output for each point in the grid
    grid_predictions = np.array([nn.predict(point) for point in grid_points])
    grid_predictions = grid_predictions.reshape(xx.shape)

    # Plot the decision boundary by assigning a color to each point in the mesh
    plt.contourf(xx, yy, grid_predictions, alpha=0.3, levels=np.linspace(0, 1, 11), cmap='coolwarm')

    # Plot training data
    for input_vector, target in zip(train_input_vectors, train_targets):
        plt.scatter(input_vector[0], input_vector[1], marker='o', color='blue' if target <= 0.5 else 'red', edgecolors='k')

    # Plot testing data
    for input_vector, target in zip(test_input_vectors, test_targets):
        plt.scatter(input_vector[0], input_vector[1], marker='x', color='blue' if target <= 0.5 else 'red')

    # Add labels and legend
    plt.xlabel('Input dimension 1')
    plt.ylabel('Input dimension 2')
    plt.title('Training and Testing Data Distribution with XOR Decision Boundary')
    # plt.legend(['Train (0)', 'Train (1)', 'Test (0)', 'Test (1)'], loc='best')
    # plt.legend(['Training Data (o)', 'Testing Data (x)'], loc='best')
    # Create custom legend
    custom_lines = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Train (0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Train (1)'),
        plt.Line2D([0], [0], marker='x', color='w', markeredgecolor='blue', markersize=10, label='Test (0)'),
        plt.Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=10, label='Test (1)')
    ]
    
    plt.legend(handles=custom_lines, loc='best')

    plt.grid(True)
    plt.show()