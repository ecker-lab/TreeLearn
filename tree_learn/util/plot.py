import numpy as np
import plotly.express as px
import pandas as pd


# plot evaluation of individual partitions
def plot_evaluation_results_segments(ax, values, fontsize, measure, y_range=[0.6, 1], y_step=10, color="#ff7f0e", x_label="segment"):
    y_pos = np.arange(len(values))
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks(y_pos, np.arange(1, 11))
    ax.set_yticks(np.arange(y_range[0], y_range[1] + 0.1, y_step))
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(measure, fontsize=fontsize)
    ax.set_ylim(y_range)
    ax.plot(values, color=color)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
    return ax


# juxtapose two point clouds (used to visualize over- and undersegmentation errors)
def juxtapose(cloud1, cloud2, label1, label2, color1='blue', color2='red', subset=10, renderer='notebook', size=1, opacity=1):
    # Subset the point clouds
    cloud1 = cloud1[::subset]
    cloud2 = cloud2[::subset]

    # Combine both clouds to calculate the overall range
    combined_cloud = np.vstack((cloud1, cloud2))

    # Determine the range for the axes
    min_val = np.min(combined_cloud, axis=0)
    max_val = np.max(combined_cloud, axis=0)

    # Separate points and labels
    labels1 = np.ones(cloud1.shape[0]) * 1  # Assign label 1 for cloud1
    labels2 = np.ones(cloud2.shape[0]) * 2  # Assign label 2 for cloud2

    # Combine both clouds again for plotting
    point_cloud = np.vstack((cloud1, cloud2))
    labels = np.hstack((labels1, labels2))
    size = np.ones(len(labels)) * size  # Set size based on the input argument

    # Convert labels to strings
    labels = np.where(labels == 1, label1, label2)

    # Create a DataFrame for Plotly Express
    df = pd.DataFrame({
        'x': point_cloud[:, 0], 
        'y': point_cloud[:, 1], 
        'z': point_cloud[:, 2], 
        'labels': labels, 
        'size': size
    })

    # Create a 3D scatter plot using Plotly Express with colors assigned to categories
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', 
                        size='size',  # Use size for the dots
                        color_discrete_sequence=[color1, color2],
                        opacity=opacity)

    # Update the layout to have a black background and equal axis scales
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="black", range=[min_val[0], max_val[0]]),
            yaxis=dict(backgroundcolor="black", range=[min_val[1], max_val[1]]),
            zaxis=dict(backgroundcolor="black", range=[min_val[2], max_val[2]])
        ),
        paper_bgcolor="black",  # Background of the paper
        plot_bgcolor="black"    # Background of the plot
    )

    # Show the plot
    fig.show(renderer=renderer)
