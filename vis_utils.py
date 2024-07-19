import numpy as np


# pink purple orange green
COLORS1 = ['#F48FB1', '#CE93D8', '#FFCC80', '#A5D6A7', '#81D4FA', '#9FA8DA']
COLORS2 = ['#EC407A', '#AB47BC', '#FFA726', '#66BB6A', '#64B5F6', '#7986CB']
COLORS3 = ['#D81B60', '#8E24AA', '#FB8C00', '#43A047', '#2196F3', '#5C6BC0']
COLORS4 = ['#DCE775', '#FFF176', '#FFD54F', '#FF8A65']
LIME200 = '#E6EE9C'
LIMEA200 = '#EEFF41'
GRAY400 = '#757575'
BLUE200 = '#81D4FA'
BLUE300 = '#64B5F6'
BLUE500 = '#2196F3'


def hex_to_rgb(hex_color):
    hex_color = hex_color.strip('#')    
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return [r / 255, g / 255, b / 255]


def show_mask(mask, ax, idx=0, linewidth=2):
    idx = idx % 6 if idx >=6 else idx
    mask_color = LIME200 if idx == -1 else COLORS1[idx]
    mask_color = hex_to_rgb(mask_color)
    mask_color.append(0.4)
    mask_color = np.array(mask_color)

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    if idx == -1:
        ax.contour(mask, levels=[0.5], colors=LIMEA200, linewidths=linewidth)    
    else:
        ax.contour(mask, levels=[0.5], colors=COLORS3[idx], linewidths=linewidth)


def show_pos_points(coords, labels, ax, idx, color=None, marker_size=375, linewidth=1.25):
    color_idx = idx % 6 if idx >=6 else idx
    
    pos_points = coords[labels==1]
    if idx != -1:
        ax.scatter(pos_points[idx, 0], pos_points[idx, 1], color=COLORS3[color_idx], marker='*', s=marker_size, edgecolor='white', linewidth=linewidth)
    else:
        if color is None:
            color = BLUE500
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color=color, marker='*', s=marker_size, edgecolor='white', linewidth=linewidth)


def show_neg_points(coords, labels, ax, color=None, marker_size=375, linewidth=1.25):
    if color is None:
        color = GRAY400
    neg_points = coords[labels==0]
    ax.scatter(neg_points[:, 0],  neg_points[:, 1], color=color, marker='*', s=marker_size, edgecolor='white', linewidth=linewidth)
