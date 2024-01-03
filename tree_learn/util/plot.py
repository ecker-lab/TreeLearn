import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

###########################################################################
######################### some plotting functionality that might be helpful
###########################################################################

def get_ptcloud_img(ptcloud,roll=0,pitch=0, point_size=1, cmap=None, savepath=None, xlims=None, ylims=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10,10), constrained_layout=True)
        ax = plt.axes(projection=Axes3D.name)
    x, y, z, c = ptcloud.transpose(1, 0)
    ax.axis('off')
    vals = np.unique(c)
    for i, val in enumerate(vals):
        c[c==val] = i
    ax.set_box_aspect((np.ptp(x), np.ptp(y) , np.ptp(z)))     
    ax.view_init(roll,pitch)
    if cmap == None:
        ax.scatter(x, y, z, zdir='z', s=point_size, cmap=generate_colormap(len(vals)))
    else:
        ax.scatter(x, y, z, c=c, s=point_size, cmap=cmap)

    if xlims is not None:
        ax.set_xlim([xlims[0], xlims[1]])
    if ylims is not None:
        ax.set_ylim([ylims[0], ylims[1]])
    ax.set_axis_off()

    if savepath is not None:
        plt.savefig(savepath, transparent=True, dpi=300)
    return fig, ax

def convert_to_pcd(path, points, color=False, seed=3):
    import open3d as o3d
    # define a color palette
    if seed is not None:
        np.random.seed(3)
    labels = np.unique(points[:, 3])
    num_drawpoints = len(points)

    if color is True:
        n_color_palette = len(labels)
        color_palette = np.random.uniform(size=(n_color_palette, 3))
        # define how labels get mapped to color palette
        color_palette_mapping = {j: i for i, j in enumerate(np.sort(labels))}
        color_palette[-1] = [0,0,0]
        colors = np.empty((num_drawpoints, 3))

    for i in range(num_drawpoints):
        ind = int(points[i][-1])
        if color is True:
            colors[i] = color_palette[color_palette_mapping[ind]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if color is True:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=True, print_progress=True)


def discrete_cmap(N, base_cmap="hsv"):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return plt.cm.colors.ListedColormap(color_list, color_list, N)


def class_sampling(cluster, num=300, return_choice=False):
    instances = np.unique(cluster[:,-1])
    ls = []
    choices = []
    for instance in instances:
        pc = cluster[cluster[:,-1] == instance]
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        choices.append(choice)
        ls.append(pc[choice])
    
    if return_choice:
        return np.concatenate(choices)
    else:
        return np.vstack(ls)


def plot_offset(cluster, ax=None, title=None, num_points=300, point_size=0.5, cmap=None, string=None, dpi=300, xlims=None, ylims=None, alpha=1): 
    if ax is  None:
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        ax = plt.axes()
    if title is not None:
        ax.title.set_text(title)
    if num_points < len(cluster):
        cluster = class_sampling(cluster, num_points)
    x, y, c = cluster[:,0], cluster[:,1], cluster[:,-1]
    #ax.axis('off')
    vals = np.unique(c)
    c_old = c.copy()
    for i, val in enumerate(vals):
        c[c_old==val] = i
    ax.set_aspect('equal', adjustable='box')  
    if cmap == None:
        ax.scatter(x, y, s=point_size,c=c, cmap=generate_colormap(len(vals) +5), alpha=alpha)
    else:
        ax.scatter(x, y, s=point_size,c=c, cmap=cmap, alpha=alpha)

    if xlims is not None:
        ax.set_xlim([xlims[0], xlims[1]])
    if ylims is not None:
        ax.set_ylim([ylims[0], ylims[1]])
    ax.set_axis_off()

    if string is not None:
        plt.savefig(string, bbox_inches="tight", transparent=True, dpi=dpi)


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80
    number_of_shades = 12
    if number_of_distinct_colors < number_of_shades:
        number_of_shades = number_of_distinct_colors
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)
    arr_by_shade_columns = arr_by_shade_rows.T
    number_of_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)


def get_centers(instance_labels, offset_labels, coords_float):
    unique_values = np.unique(instance_labels)
    first_occurrences = [np.where(instance_labels == value)[0][0] for value in unique_values]
    centers =  coords_float[first_occurrences] + offset_labels[first_occurrences]
    return centers, unique_values


def get_trees_in_area(centers, instances, center_point, radius):
    distances = np.linalg.norm(centers[:,:2] - center_point, axis=1)
    return_instances = instances[distances < radius]
    return return_instances


def get_trunk_point_of_tree(coords, instances, tree_num):
    coords_tree = coords[instances==tree_num]
    min_z = np.min(coords_tree[:, -1])
    lowest_points = coords_tree[coords_tree[:, -1] <= min_z + 0.3]
    mean = np.mean(lowest_points, 0)
    return mean[:-1]


def get_chunk_around_instance(coords, labels, inst_num, edge_length, ignore_label=None):
    x = coords[:, 0]
    y = coords[:, 1]

    x_inst = coords[labels == inst_num][:, 0]
    y_inst = coords[labels == inst_num][:, 1]
    x_inst_mean = x_inst.mean()
    y_inst_mean = y_inst.mean()

    chunk_inds1 = x <= (x_inst_mean + edge_length / 2)
    chunk_inds2 = x >= (x_inst_mean - edge_length / 2)
    chunk_inds3 = y <= (y_inst_mean + edge_length / 2)
    chunk_inds4 = y >= (y_inst_mean - edge_length / 2)
    if ignore_label is not None:
        chunk_inds = chunk_inds1 & chunk_inds2 & chunk_inds3 & chunk_inds4 & (labels != ignore_label)
    else:
        chunk_inds = chunk_inds1 & chunk_inds2 & chunk_inds3 & chunk_inds4

    return chunk_inds


def crop_and_set_bg_color(fig, bg_hex="#e6e6e6", save_path=None):
    from PIL import Image
    from io import BytesIO
    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    
    # Open the image with PIL
    img = Image.open(buf)
    
    # Get the bounding box and crop the image
    bbox = img.getbbox()
    crop_img = img.crop(bbox)
    
    # Add background color
    bg = Image.new("RGB", crop_img.size, bg_hex)
    bg.paste(crop_img, mask=crop_img)
    
    # If a save path is provided, save the image to this path
    if save_path:
        bg.save(save_path, format="png")
        return bg
    else:
        # Otherwise, save the modified image to a BytesIO object and return this
        out_buf = BytesIO()
        bg.save(out_buf, format="png")
        out_buf.seek(0)
        return out_buf


def plot_instance_evaluation_segments(ax, values, fontsize, measure, y_range=[0.6, 1], y_step=10, color="#ff7f0e", x_label="segment"):
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
