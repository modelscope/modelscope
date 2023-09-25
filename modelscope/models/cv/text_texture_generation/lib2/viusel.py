import os
import sys

import imageio.v2 as imageio
# visualization
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from modelscope.models.cv.text_texture_generation.lib2.camera import \
    polar_to_xyz
from modelscope.models.cv.text_texture_generation.lib2.init_view import *

matplotlib.use('Agg')

sys.path.append('.')


def visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_score,
                        device):
    quad_mask_tensor = quad_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    quad_mask_image_tensor = torch.zeros_like(quad_mask_tensor)

    for idx in PALETTE:
        selected = quad_mask_tensor[quad_mask_tensor == idx].reshape(-1, 3)
        selected = torch.FloatTensor(
            PALETTE[idx]).to(device).unsqueeze(0).repeat(selected.shape[0], 1)

        quad_mask_image_tensor[quad_mask_tensor == idx] = selected.reshape(-1)

    quad_mask_image_np = quad_mask_image_tensor[0].cpu().numpy().astype(
        np.uint8)
    quad_mask_image = Image.fromarray(quad_mask_image_np).convert('RGB')
    quad_mask_image.save(
        os.path.join(mask_image_dir,
                     '{}_quad_{:.5f}.png'.format(view_idx, view_score)))


def visualize_outputs(output_dir, init_image_dir, mask_image_dir,
                      inpainted_image_dir, num_views):
    # subplot settings
    num_col = 3
    num_row = 1
    sus = 4

    summary_image_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_image_dir, exist_ok=True)

    # graph settings
    print('=> visualizing results...')
    for view_idx in range(num_views):
        plt.switch_backend('agg')
        fig = plt.figure(dpi=100)
        fig.set_size_inches(sus * num_col, sus * (num_row + 1))
        fig.set_facecolor('white')

        # rendering
        plt.subplot2grid((num_row, num_col), (0, 0))
        plt.imshow(
            Image.open(
                os.path.join(init_image_dir, '{}.png'.format(view_idx))))
        plt.text(
            0,
            0,
            'Rendering',
            fontsize=16,
            color='black',
            backgroundcolor='white')
        plt.axis('off')

        # mask
        plt.subplot2grid((num_row, num_col), (0, 1))
        plt.imshow(
            Image.open(
                os.path.join(mask_image_dir,
                             '{}_project.png'.format(view_idx))))
        plt.text(
            0,
            0,
            'Project Mask',
            fontsize=16,
            color='black',
            backgroundcolor='white')
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

        # inpainted
        plt.subplot2grid((num_row, num_col), (0, 2))
        plt.imshow(
            Image.open(
                os.path.join(inpainted_image_dir, '{}.png'.format(view_idx))))
        plt.text(
            0,
            0,
            'Inpainted',
            fontsize=16,
            color='black',
            backgroundcolor='white')
        plt.axis('off')

        plt.savefig(
            os.path.join(summary_image_dir, '{}.png'.format(view_idx)),
            bbox_inches='tight')
        fig.clf()

    # generate GIF
    images = [
        imageio.imread(
            os.path.join(summary_image_dir, '{}.png'.format(view_idx)))
        for view_idx in range(num_views)
    ]
    imageio.mimsave(
        os.path.join(summary_image_dir, 'output.gif'), images, duration=1)

    print('=> done!')


def visualize_principle_viewpoints(output_dir, dist_list, elev_list,
                                   azim_list):
    theta_list = [e for e in azim_list]
    phi_list = [90 - e for e in elev_list]
    DIST = dist_list[0]

    xyz_list = [
        polar_to_xyz(theta, phi, DIST)
        for theta, phi in zip(theta_list, phi_list)
    ]

    xyz_np = np.array(xyz_list)
    color_np = np.array([[0, 0, 0]]).repeat(xyz_np.shape[0], 0)

    ax = plt.axes(projection='3d')
    SCALE = 0.8
    ax.set_xlim((-DIST, DIST))
    ax.set_ylim((-DIST, DIST))
    ax.set_zlim((-SCALE * DIST, SCALE * DIST))

    ax.scatter(
        xyz_np[:, 0],
        xyz_np[:, 2],
        xyz_np[:, 1],
        s=100,
        c=color_np,
        depthshade=True,
        label='Principle views')
    ax.scatter([0], [0], [0],
               c=[[1, 0, 0]],
               s=100,
               depthshade=True,
               label='Object center')

    # draw hemisphere
    # theta inclination angle
    # phi azimuthal angle
    n_theta = 50  # number of values for theta
    n_phi = 200  # number of values for phi
    r = DIST  # radius of sphere

    # theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
    theta, phi = np.mgrid[0.0:1 * np.pi:n_theta * 1j,
                          0.0:2.0 * np.pi:n_phi * 1j]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.25, linewidth=1)

    # Make the grid
    ax.quiver(
        xyz_np[:, 0],
        xyz_np[:, 2],
        xyz_np[:, 1],
        -xyz_np[:, 0],
        -xyz_np[:, 2],
        -xyz_np[:, 1],
        normalize=True,
        length=0.3)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.view_init(30, 35)
    ax.legend()

    plt.show()

    plt.savefig(os.path.join(output_dir, 'principle_viewpoints.png'))


def visualize_refinement_viewpoints(output_dir, selected_view_ids, dist_list,
                                    elev_list, azim_list):
    theta_list = [azim_list[i] for i in selected_view_ids]
    phi_list = [90 - elev_list[i] for i in selected_view_ids]
    DIST = dist_list[0]

    xyz_list = [
        polar_to_xyz(theta, phi, DIST)
        for theta, phi in zip(theta_list, phi_list)
    ]

    xyz_np = np.array(xyz_list)
    color_np = np.array([[0, 0, 0]]).repeat(xyz_np.shape[0], 0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    SCALE = 0.8
    ax.set_xlim((-DIST, DIST))
    ax.set_ylim((-DIST, DIST))
    ax.set_zlim((-SCALE * DIST, SCALE * DIST))

    ax.scatter(
        xyz_np[:, 0],
        xyz_np[:, 2],
        xyz_np[:, 1],
        c=color_np,
        depthshade=True,
        label='Refinement views')
    ax.scatter([0], [0], [0],
               c=[[1, 0, 0]],
               s=100,
               depthshade=True,
               label='Object center')

    # draw hemisphere
    # theta inclination angle
    # phi azimuthal angle
    n_theta = 50  # number of values for theta
    n_phi = 200  # number of values for phi
    r = DIST  # radius of sphere

    # theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
    theta, phi = np.mgrid[0.0:1 * np.pi:n_theta * 1j,
                          0.0:2.0 * np.pi:n_phi * 1j]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.25, linewidth=1)

    # Make the grid
    ax.quiver(
        xyz_np[:, 0],
        xyz_np[:, 2],
        xyz_np[:, 1],
        -xyz_np[:, 0],
        -xyz_np[:, 2],
        -xyz_np[:, 1],
        normalize=True,
        length=0.3)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.view_init(30, 35)
    ax.legend()

    plt.show()

    plt.savefig(os.path.join(output_dir, 'refinement_viewpoints.png'))

    fig.clear()
