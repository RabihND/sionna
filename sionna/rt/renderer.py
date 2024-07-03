#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing-based renderer.
"""

import drjit as dr
import matplotlib
import mitsuba as mi
import numpy as np
import tensorflow as tf

from .camera import Camera
from .utils import paths_to_segments, scene_scale, mitsuba_rectangle_to_world


def render(scene, camera, paths, show_paths, show_devices, num_samples,
           resolution, fov,
           coverage_map=None, cm_tx=0, cm_db_scale=True,
           cm_vmin=None, cm_vmax=None):
    r"""
    Renders two images with path tracing:
    1. Base scene with the meshes
    2. Paths, radio devices and coverage map,
    then composites them together.

    We adopt this approach because as of the time of writing, Mitsuba
    does not support adding or removing objects from a scene after
    it has been loaded.

    Input
    ------
    camera : str | :class:`~sionna.rt.Camera`
        The name or instance of a :class:`~sionna.rt.Camera`

    paths : :class:`~sionna.rt.Paths` | `None`
        Simulated paths generated by
        :meth:`~sionna.rt.Scene.compute_paths()` or `None`.
        If `None`, only the scene is rendered.

    show_paths : bool
        If `paths` is not `None`, shows the paths.

    show_devices : bool
        If `paths` is not `None`, shows the radio devices.

    coverage_map : :class:`~sionna.rt.CoverageMap` | `None`
        An optional coverage map to overlay in the scene for visualization.
        Defaults to `None`.

    cm_tx : int | str
        When `coverage_map` is specified, controls which of the transmitters
        to display the coverage map for. Either the transmitter's name
        or index can be given.
        Defaults to `0`.

    cm_db_scale: bool
        Use logarithmic scale for coverage map visualization, i.e. the
        coverage values are mapped with:
        :math:`y = 10 \cdot \log_{10}(x)`.
        Defaults to `True`.

    cm_vmin, cm_vmax: floot | None
        For coverage map visualization, defines the range of path gains that
        the colormap covers.
        If set to None, then covers the complete range.
        Defaults to `None`.

    num_samples : int
        Number of rays thrown per pixel.

    resolution : [2], int
        Size of the rendered figure.

    fov : float
        Field of view, in degrees.

    Output
    -------
    : :class:`~mitsuba.Bitmap`
        Rendered image
    """
    s1 = scene.mi_scene
    sensor = make_render_sensor(scene, camera, resolution=resolution, fov=fov)

    integrator = mi.load_dict({
        'type': 'path',
        'max_depth': 8,
        'hide_emitters': True,
    })
    img1 = mi.render(s1, sensor=sensor, integrator=integrator, spp=num_samples)
    img1 = img1.numpy()

    needs_compositing = (
        (show_paths and paths is not None)
        or (coverage_map is not None)
        or show_devices
    )
    if needs_compositing:
        if coverage_map is not None:
            coverage_map = _coverage_map_to_textured_rectangle(
                coverage_map, tx=cm_tx, db_scale=cm_db_scale,
                vmin=cm_vmin, vmax=cm_vmax,
                viewpoint=sensor.world_transform().translation())

        s2 = results_to_mitsuba_scene(scene, paths=paths,
                                      show_paths=show_paths,
                                      show_devices=show_devices,
                                      coverage_map=coverage_map)
        depth_integrator = mi.load_dict({
            'type': 'depth'
        })

        depth1 = mi.render(s1, sensor=sensor, integrator=depth_integrator,
                           spp=num_samples)
        depth1 = unmultiply_alpha(depth1.numpy())

        img2 = mi.render(s2, sensor=sensor, integrator=integrator,
                         spp=num_samples)
        img2 = img2.numpy()
        depth2 = mi.render(s2, sensor=sensor, integrator=depth_integrator,
                           spp=num_samples)
        depth2 = unmultiply_alpha(depth2.numpy())

        # Alpha compositing using the renderings (stored as pre-multiplied
        # alpha)
        alpha1 = img1[:, :, 3]
        alpha2 = img2[:, :, 3]
        composite = img2 + img1 * (1 - alpha2[:, :, None])

        # Use the composite only in unoccluded regions (based on depth info)
        # TODO: can probably do a nicer transition based on depth values
        img3 = np.where(
            (alpha1[:, :, None] > 0) & (depth1 < depth2),
            img1,
            composite
        )
        img3[:, :, 3] = np.maximum(img1[:, :, 3], composite[:, :, 3])

    else:
        # No need for any compositing, we just have to show the scene as-is
        img3 = img1

    return mi.Bitmap(img3)

def make_render_sensor(scene, camera, resolution, fov):
    r"""
    Instantiates a Mitsuba sensor (camera) from the provided ``camera`` object.

    Input
    ------
    scene : :class:`~sionna.rt.Scene`
        The scene

    camera : str | :class:`~sionna.rt.Camera` | :class:`~mitsuba.Sensor`
        A camera

    resolution : [2], int
        Size of the rendered figure.

    fov : float
        Field of view, in degrees.

    Output
    -------
    : :class:`~mitsuba.Sensor`
        A Mitsuba sensor (camera)
    """
    props = {
        'type': 'perspective',
    }

    if isinstance(camera, str):

        if camera == 'preview':
            # Use the viewpoint from the preview.
            w = scene.preview_widget
            if w is None:
                raise RuntimeError("Could not find an open preview widget, "
                                   "please call `scene.preview()` first.")

            cam = w.camera
            props['to_world'] = mi.ScalarTransform4f.look_at(
                origin=cam.position,
                target=w.orbit.target,
                up=(0, 0, 1),
            )
            props['near_clip'] = cam.near
            props['far_clip'] = cam.far
            del w, cam

        else:
            cam_name = camera
            camera = scene.get(cam_name)
            if not isinstance(camera, Camera):
                raise ValueError(f"The scene has no camera named '{cam_name}'")

    if isinstance(camera, Camera):
        world_transform = camera.world_transform.matrix.numpy()
        props['to_world'] = mi.ScalarTransform4f(world_transform)
        props['near_clip'] = 0.1
        props['far_clip'] = 10000

    elif isinstance(camera, mi.Sensor):
        sensor_params = mi.traverse(camera)
        world_transform = camera.world_transform().matrix.numpy()
        props['to_world'] = mi.ScalarTransform4f(world_transform)
        props['near_clip'] = sensor_params['near_clip']
        props['far_clip'] = sensor_params['far_clip']

    elif isinstance(camera, str):
        # Do nothing as this was already handled. This is to avoid wrongly
        # raising an exception
        pass

    else:
        raise ValueError(f'Unsupported camera type: {type(camera)}')

    if fov is not None:
        props['fov'] = fov
        props['fov_axis'] = 'x'
    props['film'] = {
        'type': 'hdrfilm',
        'width': resolution[0],
        'height': resolution[1],
        'pixel_format': 'rgba',
        'rfilter': {'type': 'box'},
    }
    return mi.load_dict(props)


def results_to_mitsuba_scene(scene, paths, show_paths, show_devices,
                             coverage_map=None):
    """
    Builds a Mitsuba scene with only the paths

    Input
    -----
    scene : :class:`~sionna.rt.Scene`
        The scene

    paths : :class:`~sionna.rt.Paths` | `None`
        Simulated paths generated by
        :meth:`~sionna.rt.Scene.compute_paths()` or `None`.
        If `None`, only the scene is rendered.
        Defaults to `None`.

    show_paths : bool
        If `paths` is not `None`, shows the paths.
        Defaults to `True`.

    show_devices : bool
        If `paths` is not `None`, shows the radio devices.
        Defaults to `True`.

    coverage_map : dict
        Dictionary describing a Mitsuba rectangle shape, with a
        textured emitter showing the coverage map to display.
        See `coverage_map_to_textured_rectangle`.

    Output
    -------
    : :class:`~mitsuba.Scene`
        A Mitsuba scene
    """
    objects = {
        'type': 'scene',
    }
    sc, tx_positions, rx_positions, ris_positions, _ = scene_scale(scene)
    ris_orientations = [ris.orientation for ris in scene.ris.values()]
    ris_sizes = [ris.size for ris in scene.ris.values()]
    transmitter_colors = [transmitter.color.numpy() for
                          transmitter in scene.transmitters.values()]
    receiver_colors = [receiver.color.numpy() for
                       receiver in scene.receivers.values()]
    ris_colors = [ris.color.numpy() for
                           ris in scene.ris.values()]

    # --- Radio devices, shown as spheres
    if show_devices:
        radius = max(0.0025 * sc, 1)
        for source, color in ((tx_positions, transmitter_colors),
                              (rx_positions, receiver_colors)):
            for index, (k, p) in enumerate(source.items()):
                key = 'rd-' + k
                assert key not in objects
                objects[key] = {
                    'type': 'sphere',
                    'center': p,
                    'radius': radius,
                    'light': {
                        'type': 'area',
                        'radiance': {'type': 'rgb', 'value': color[index]},
                    },
                }

    # --- RIS, shown as rectangles
    if show_devices:
        for k, o, s, c in zip(ris_positions, ris_orientations, ris_sizes,
                              ris_colors):
            p = tf.constant(ris_positions[k])
            key = 'ris-' + k
            assert key not in objects
            to_world = mitsuba_rectangle_to_world(p, o, s, ris=True)
            objects[key] = {
                'type': 'rectangle',
                'to_world': to_world,
                'light': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': c},
                },
            }

    # --- Paths, shown as cylinders (the closest we have to lines)
    if (paths is not None) and show_paths:
        path_color = [0.75, 0.75, 0.75]
        starts, ends = paths_to_segments(paths)

        for i, (s, e) in enumerate(zip(starts, ends)):
            # TODO: avoid the shearing warning
            objects[f'path-{i:06d}'] = {
                'type': 'cylinder',
                'p0': s,
                'p1': e,
                'radius': 0.1,
                'light': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': path_color},
                },
            }

    if coverage_map is not None:
        objects['coverage-map'] = coverage_map

    # Temporarily raise log level to silence warning about cylinders shearing
    # TODO: shouldn't need this, maybe the warning is too sensitive
    logger = mi.Thread.thread().logger()
    level = logger.log_level()
    logger.set_log_level(mi.LogLevel.Error)
    new_scene = mi.load_dict(objects)
    logger.set_log_level(level)

    return new_scene


def _coverage_map_to_textured_rectangle(coverage_map, tx=0, db_scale=True,
                                        vmin=None, vmax=None,
                                        viewpoint=None):
    to_world = coverage_map.to_world()
    # Resample values from cell centers to cell corners
    coverage_map = resample_to_corners(
        coverage_map[tx, :, :].numpy().squeeze()
    )

    texture, opacity = _coverage_map_texture(
        coverage_map, db_scale=db_scale, vmin=vmin, vmax=vmax)
    bsdf = {
        'type': 'mask',
        'opacity': {
            'type': 'bitmap',
            'bitmap': mi.Bitmap(opacity.astype(np.float32)),
        },
        'nested': {
            'type': 'diffuse',
            'reflectance': 0.,
        },
    }

    emitter = {
        'type': 'area',
        'radiance': {
            'type': 'bitmap',
            'bitmap': mi.Bitmap(texture.astype(np.float32)),
        },
    }

    flip_normal = False
    if viewpoint is not None:
        # Area emitters are single-sided, so we need to flip the rectangle's
        # normals if the camera is on the wrong side.
        p0 = to_world.transform_affine([-1, -1, 0])
        p1 = to_world.transform_affine([-1, 0, 0])
        p2 = to_world.transform_affine([0, -1, 0])
        plane_center = to_world.transform_affine([0, 0, 0])
        normal = dr.cross(p1 - p0, p2 - p0)
        flip_normal = dr.dot(plane_center - viewpoint.numpy(), normal) < 0

    return {
        'type': 'rectangle',
        'flip_normals': flip_normal,
        'to_world': to_world,
        'bsdf': bsdf,
        'emitter': emitter,
    }


def coverage_map_color_mapping(coverage_map, db_scale=True,
                               vmin=None, vmax=None):
    """
    Prepare a Matplotlib color maps and normalizing helper based on the
    requested value scale to be displayed.
    Also applies the dB scaling to a copy of the coverage map, if requested.
    """
    valid = np.logical_and(coverage_map > 0., np.isfinite(coverage_map))
    coverage_map = coverage_map.copy()
    if db_scale:
        coverage_map[valid] = 10. * np.log10(coverage_map[valid])
    else:
        coverage_map[valid] = coverage_map[valid]

    if vmin is None:
        vmin = coverage_map[valid].min()
    if vmax is None:
        vmax = coverage_map[valid].max()
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.colormaps.get_cmap('viridis')
    return coverage_map, normalizer, color_map


def resample_to_corners(values):
    """
    Takes a 2D NumPy array of values defined at cell centers and converts it
    into an array (with one more row and one more column) with values defined at
    cell corners.
    """
    assert values.ndim == 2
    padded = np.pad(values, pad_width=((1, 1), (1, 1)), mode='edge')
    return 0.25 * (
          padded[ :-1,  :-1]
        + padded[1:  ,  :-1]
        + padded[ :-1, 1:  ]
        + padded[1:  , 1:  ]
    )


def _coverage_map_texture(coverage_map, db_scale=True, vmin=None, vmax=None):
    # Leave zero-valued regions as transparent
    valid = coverage_map > 0.
    opacity = valid.astype(np.float32)

    # Color mapping of real values
    coverage_map, normalizer, color_map = coverage_map_color_mapping(
        coverage_map, db_scale=db_scale, vmin=vmin, vmax=vmax)
    texture = color_map(normalizer(coverage_map))[:, :, :3]
    # Colors from the color map are gamma-compressed, go back to linear
    texture = np.power(texture, 2.2)

    # Pre-multiply alpha to avoid fringe
    texture *= opacity[:, :, None]

    return texture, opacity


def unmultiply_alpha(arr):
    """
    De-multiply the alpha channel

    Input
    -----
    arr : [w,h,4]
        An image

    Output
    -------
    arr : [w,h,4]
        Image with the alpha channel de-multiplied.
    """
    arr = arr.copy()
    alpha = arr[:, :, 3]
    weight = 1. / np.where(alpha > 0, alpha, 1.)
    arr[:, :, :3] *= weight[:, :, None]
    return arr
