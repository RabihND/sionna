#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class that stores coverage map
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import tensorflow as tf
from sionna.utils import expand_to_rank, log10
from .utils import rotation_matrix, mitsuba_rectangle_to_world
import warnings

class CoverageMap:
    # pylint: disable=line-too-long
    r"""
    CoverageMap()

    Stores the simulated coverage maps

    A coverage map is generated for the loaded scene for every transmitter using
    :meth:`~sionna.rt.Scene.coverage_map`. Please refer to the documentation of this function
    for further details.

    An instance of this class can be indexed like a tensor of rank three with
    shape ``[num_tx, num_cells_y, num_cells_x]``, i.e.:

    .. code-block:: Python

        cm = scene.coverage_map()
        print(cm[0])      # prints the coverage map for transmitter 0
        print(cm[0,1,2])  # prints the value of the cell (1,2) for transmitter 0

    where ``scene`` is the :class:`~sionna.rt.Scene` loaded using
    :func:`~sionna.rt.load_scene`.

    Example
    -------
    .. code-block:: Python

        import sionna
        from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
        scene = load_scene(sionna.rt.scene.munich)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=8,
                                  num_cols=2,
                                  vertical_spacing=0.7,
                                  horizontal_spacing=0.5,
                                  pattern="tr38901",
                                  polarization="VH")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="dipole",
                                  polarization="cross")
        # Add a transmitters
        tx = Transmitter(name="tx",
                      position=[8.5,21,30],
                      orientation=[0,0,0])
        scene.add(tx)
        tx.look_at([40,80,1.5])

        # Compute coverage map
        cm = scene.coverage_map(max_depth=8)

        # Show coverage map
        cm.show()

    .. figure:: ../figures/coverage_map_show.png
        :align: center
    """

    def __init__(self,
                 center,
                 orientation,
                 size,
                 cell_size,
                 value,
                 scene,
                 dtype=tf.complex64):

        self._rdtype = dtype.real_dtype

        if (tf.rank(center) != 1) or (tf.shape(center)[0] != 3):
            msg = "`center` must be shaped as [x,y,z] (rank=1 and shape=[3])"
            raise ValueError(msg)

        if (tf.rank(orientation) != 1) or (tf.shape(orientation)[0] != 3):
            msg = "`orientation` must be shaped as [a,b,c]"\
                  " (rank=1 and shape=[3])"
            raise ValueError(msg)

        if (tf.rank(size) != 1) or (tf.shape(size)[0] != 2):
            msg = "`size` must be shaped as [w,h]"\
                  " (rank=1 and shape=[2])"
            raise ValueError(msg)

        if (tf.rank(cell_size) != 1) or (tf.shape(cell_size)[0] != 2):
            msg = "`cell_size` must be shaped as [w,h]"\
                  " (rank=1 and shape=[2])"
            raise ValueError(msg)

        num_cells_x = tf.cast(tf.math.ceil(size[0]/cell_size[0]), tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(size[1]/cell_size[1]), tf.int32)

        if (tf.rank(value) != 3)\
            or (tf.shape(value)[1] != num_cells_y)\
            or (tf.shape(value)[2] != num_cells_x):
            msg = "`value` must have shape"\
                  " [num_tx, num_cells_y, num_cells_x]"
            raise ValueError(msg)

        self._center = tf.cast(center, self._rdtype)
        self._orientation = tf.cast(orientation, self._rdtype)
        self._size = tf.cast(size, self._rdtype)
        self._cell_size = tf.cast(cell_size, self._rdtype)
        self._value = tf.cast(value, self._rdtype)
        self._transmitters = scene.transmitters
        # print(f"Transmitters: {self._transmitters}")
        # Dict mapping names to index for transmitters
        self._tx_name_2_ind = {}
        for tx_ind, tx_name in enumerate(self._transmitters):
            self._tx_name_2_ind[tx_name] = tx_ind

        ###############################################################
        # Position of the center of the cells in the world
        # coordinate system
        ###############################################################
        # [num_cells_x]
        x_positions = tf.range(num_cells_x, dtype=self._rdtype)
        x_positions = (x_positions + 0.5)*self._cell_size[0]
        # [num_cells_x, num_cells_y]
        x_positions = tf.expand_dims(x_positions, axis=1)
        x_positions = tf.tile(x_positions, [1, num_cells_y])
        # [num_cells_y]
        y_positions = tf.range(num_cells_y, dtype=self._rdtype)
        y_positions = (y_positions + 0.5)*self._cell_size[1]
        # [num_cells_x, num_cells_y]
        y_positions = tf.expand_dims(y_positions, axis=0)
        y_positions = tf.tile(y_positions, [num_cells_x, 1])
        # [num_cells_x, num_cells_y, 2]
        cell_pos = tf.stack([x_positions, y_positions], axis=-1)
        # Move to global coordinate system
        # [1, 1, 2]
        size = expand_to_rank(self._size, tf.rank(cell_pos), 0)
        # [num_cells_x, num_cells_y, 2]
        cell_pos = cell_pos - size*0.5
        # [num_cells_x, num_cells_y, 3]
        cell_pos = tf.concat([cell_pos,
                                tf.zeros([num_cells_x, num_cells_y, 1],
                                        dtype=self._rdtype)],
                                axis=-1)
        # [3, 3]
        rot_cm_2_gcs = rotation_matrix(self._orientation)
        # [1, 1, 3, 3]
        rot_cm_2_gcs_ = expand_to_rank(rot_cm_2_gcs, tf.rank(cell_pos)+1,
                                        axis=0)
        # [num_cells_x, num_cells_y, 3]
        cell_pos = tf.linalg.matvec(rot_cm_2_gcs_, cell_pos)
        # [num_cells_x, num_cells_y, 3]
        cell_pos = cell_pos + self._center
        # [num_cells_y, num_cells_x, 3]
        cell_pos = tf.transpose(cell_pos, [1,0,2])
        self._cell_pos = cell_pos

        ######################################################################
        # Position of the transmitters, receivers, and RIS in the coverage map
        ######################################################################
        # [num_tx/num_rx/num_ris, 3]
        tx_pos = [tx.position for tx in scene.transmitters.values()]
        tx_orientation = [tx.orientation for tx in scene.transmitters.values()]
        tx_pos = tf.stack(tx_pos, axis=0)

        rx_pos = [rx.position for rx in scene.receivers.values()]
        rx_pos = tf.stack(rx_pos, axis=0)
        if len(rx_pos)==0:
            rx_pos = tf.zeros([0,3], dtype=self._rdtype)

        ris_pos = [ris.position for ris in scene.ris.values()]
        ris_pos = tf.stack(ris_pos, axis=0)
        if len(ris_pos)==0:
            ris_pos = tf.zeros([0,3], dtype=self._rdtype)

        # [num_tx/num_rx/num_ris, 3]
        center_ = tf.expand_dims(self._center, axis=0)
        tx_pos = tx_pos - center_
        rx_pos = rx_pos - center_
        ris_pos = ris_pos - center_

        # [3, 3]
        rot_gcs_2_cm = tf.transpose(rot_cm_2_gcs)
        # [1, 3, 3]
        rot_gcs_2_cm_ = tf.expand_dims(rot_gcs_2_cm, axis=0)
        # Positions in the coverage map system
        # [num_tx/num_rx/num_ris, 3]
        tx_pos = tf.linalg.matvec(rot_gcs_2_cm_, tx_pos)
        rx_pos = tf.linalg.matvec(rot_gcs_2_cm_, rx_pos)
        ris_pos = tf.linalg.matvec(rot_gcs_2_cm_, ris_pos)

        # Keep only x and y
        # [num_tx/num_rx/num_ris, 2]
        tx_pos = tx_pos[:,:2]
        rx_pos = rx_pos[:,:2]
        ris_pos = ris_pos[:,:2]

        # Using the bottom left corner as origin
        # [num_tx/num_rx/num_ris, 2]
        tx_pos = tx_pos + self._size*0.5
        rx_pos = rx_pos + self._size*0.5
        ris_pos = ris_pos + self._size*0.5
        # Quantizing
        # [num_tx/num_rx/num_ris, 2]
        tx_pos = tf.cast(tf.math.floor(tx_pos/self._cell_size), tf.int32)
        rx_pos = tf.cast(tf.math.floor(rx_pos/self._cell_size), tf.int32)
        ris_pos = tf.cast(tf.math.floor(ris_pos/self._cell_size), tf.int32)

        self._tx_pos = tx_pos
        self._tx_orientation = tx_orientation
        self._rx_pos = rx_pos
        self._ris_pos = ris_pos

    @property
    def center(self):
        """
        [3], tf.float : Get the center of the coverage map
        """
        return self._center

    @property
    def orientation(self):
        """
        [3], tf.float : Get the orientation of the coverage map
        """
        return self._orientation

    @property
    def size(self):
        """
        [2], tf.float : Get the size of the coverage map
        """
        return self._size

    @property
    def cell_size(self):
        """
        [2], tf.float : Get the resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            of the cells of the coverage map
        """
        return self._cell_size

    @property
    def cell_centers(self):
        """
        [num_cells_y, num_cells_x, 3], tf.float : Get the positions of the
        centers of the cells in the global coordinate system
        """
        return self._cell_pos

    @property
    def num_cells_x(self):
        """
        int : Get the number of cells along the local X-axis
        """
        return self._value.shape[2]

    @property
    def num_cells_y(self):
        """
        int : Get the number of cells along the local Y-axis
        """
        return self._value.shape[1]

    @property
    def num_tx(self):
        """
        int : Get the number of transmitters
        """
        return self._value.shape[0]

    def as_tensor(self):
        """
        Returns the coverage map as a tensor

        Output
        ------
        : [num_tx, num_cells_y, num_cells_x], tf.float
            The coverage map as a tensor
        """
        return self._value

    def show(self, tx=0, vmin=None, vmax=None,
              show_tx=True,show_tx_orientation=False, show_rx=False, show_ris=False):
        r"""show(tx=0, vmin=None, vmax=None, show_tx=True)

        Visualizes a coverage map

        The position of the transmitter is indicated by a red "+" marker.
        The positions of the receivers are indicated by blue "x" markers.
        The positions of the RIS are indicated by black "*" markers.

        Input
        -----
        tx : int | str
            Index or name of the transmitter for which to show the coverage map
            Defaults to 0.

        vmin,vmax : float | `None`
            Define the range of path gains that the colormap covers.
            If set to `None`, then covers the complete range.
            Defaults to `None`.

        show_tx : bool
            If set to `True`, then the position of the transmitter is shown.
            Defaults to `True`.

        show_tx_orientation : bool
            If set to `True`, then the orientation of the transmitter is shown. 
            Defaults to `False`.

        show_rx : bool
            If set to `True`, then the position of the receivers is shown.
            Defaults to `False`.

        show_ris : bool
            If set to `True`, then the position of the RIS is shown.
            Defaults to `False`.

        Output
        ------
        : :class:`~matplotlib.pyplot.Figure`
            Figure showing the coverage map
        """

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
        elif isinstance(tx, str):
            if tx in self._tx_name_2_ind:
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        else:
            raise ValueError("Invalid type for `tx`: Must be a string or int")

        # Catch expected div-by-zero warnings
        with warnings.catch_warnings(record=True) as _:
            cm = 10.*np.log10(self[tx].numpy())

        # Position of the transmitter

        # Visualization the coverage map
        fig = plt.figure()
        plt.imshow(cm, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Path gain [dB]')
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        # Visualizing transmitter, receiver, RIS positions
        if show_tx:
            tx_pos = self._tx_pos[tx]
            fig.axes[0].scatter(*tx_pos, marker='P', c='r')

            if show_tx_orientation:
                # tx_orientation tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32)
                tx_orientation = self._tx_orientation[tx]
                # Calculate the direction vector for the arrow tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32) in rad
                dir_x = np.cos(tx_orientation[0])*np.cos(tx_orientation[1])
                dir_y = np.sin(tx_orientation[0])*np.cos(tx_orientation[1])
                fig.axes[0].arrow(tx_pos[0], tx_pos[1], dir_x, dir_y, head_width=0.1, head_length=0.1, fc='r', ec='r')
                

        if show_rx:
            for rx_pos in self._rx_pos:
                fig.axes[0].scatter(*rx_pos, marker='x', c='b')

        if show_ris:
            for ris_pos in self._ris_pos:
                fig.axes[0].scatter(*ris_pos, marker='*', c='k')

        return fig

    def sample_positions(self, batch_size, tx=0, min_gain_db=None,
                         max_gain_db=None, min_dist=None, max_dist=None,
                         center_pos=False):
        # pylint: disable=line-too-long
        r"""Sample random user positions from a coverage map

        For a given coverage map, ``batch_size`` random positions are sampled
        such that the *expected*  path gain of this position is larger
        than a given threshold ``min_gain_db`` or smaller than ``max_gain_db``,
        respectively.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random positions to the transmitter ``tx``.

        Note that due to the quantization of the coverage map into cells it is
        not guaranteed that all above parameters are exactly fulfilled for a
        returned position. This stems from the fact that every
        individual cell of the coverage map describes the expected *average*
        behavior of the surface within this cell. For instance, it may happen
        that half of the selected cell is shadowed and, thus, no path to the
        transmitter exists but the average path gain is still larger than the
        given threshold. Please use ``center_pos`` = `True` to sample only
        positions from the cell centers.

        .. figure:: ../figures/cm_user_sampling.png
            :align: center

        The above figure shows an example for random positions between 220m and
        250m from the transmitter and a ``max_gain_db`` of -100 dB.
        Keep in mind that the transmitter can have a different height than the
        coverage map which also contributes to this distance.
        For example if the transmitter is located 20m above the surface of the
        coverage map and a ``min_dist`` of 20m is selected, also positions
        directly below the transmitter are sampled.

        Input
        -----
        batch_size: int
            Number of returned random positions

        min_gain_db: float | None
            Minimum path gain [dB]. Positions are only sampled from cells where
            the path gain is larger or equal to this value.
            Ignored if `None`.
            Defaults to `None`.

        max_gain_db: float | None
            Maximum path gain [dB]. Positions are only sampled from cells where
            the path gain is smaller or equal to this value.
            Ignored if `None`.
            Defaults to `None`.

        min_dist: float | None
            Minimum distance [m] from transmitter for all random positions.
            Ignored if `None`.
            Defaults to `None`.

        max_dist: float | None
            Maximum distance [m] from transmitter for all random positions.
            Ignored if `None`.
            Defaults to `None`.

        tx : int | str
            Index or name of the transmitter from whose coverage map
            positions are sampled

        center_pos: bool
            If `True`, all returned positions are sampled from the cell center
            (i.e., the grid of the coverage map). Otherwise, the positions are
            randomly drawn from the surface of the cell.
            Defaults to `False`.

        Output
        ------
        : [batch_size, 3], tf.float
            Random positions :math:`(x,y,z)` [m] that are in cells fulfilling the
            above constraints w.r.t. distance and path gain
        """

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
            tx_pos = list(self._transmitters.values())[tx].position
        elif isinstance(tx, str):
            if tx in self._tx_name_2_ind:
                tx_pos = self._transmitters[tx].position
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        else:
            raise ValueError("Invalid type for `tx`: Must be a string or int")

        # allow float values for batch_size
        if not isinstance(batch_size, (int, float)) or not batch_size%1==0:
            raise ValueError("batch_size must be int.")

        if min_gain_db is None:
            min_gain_db = -1. * np.infty
        min_gain_db = tf.constant(min_gain_db, self._rdtype)

        if max_gain_db is None:
            max_gain_db = np.infty
        max_gain_db = tf.constant(max_gain_db, self._rdtype)

        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db cannot be larger than max_gain_db.")

        if min_dist is None:
            min_dist = 0.
        min_dist = tf.constant(min_dist, self._rdtype)

        if max_dist is None:
            max_dist = np.infty
        max_dist = tf.constant(max_dist, self._rdtype)

        if min_dist > max_dist:
            raise ValueError("min_dist cannot be larger than max_dist.")

        cell_centers = self.cell_centers

        # Translate cm from lin. to dB scale
        cm_db = 10.*log10(self._value[tx, :, :])

        # Set min and max distance
        tx_pos = tf.cast(tf.reshape(tx_pos, (1,1,3)), dtype=self._rdtype)
        d = tf.math.reduce_euclidean_norm(cell_centers - tx_pos, axis=2)
        cm_inf = tf.constant(-1. * np.infty, shape=(1,1), dtype=self._rdtype)
        cm_inf = tf.tile(cm_inf, cm_db.shape)
        cm_db = tf.where(d < min_dist, cm_inf, cm_db) # min dist
        cm_db = tf.where(d > max_dist, cm_inf, cm_db) # max dist

        # Get all indices of positions with large enough path_gain
        idx = tf.where(tf.math.logical_and(cm_db > min_gain_db,
                                           cm_db < max_gain_db))

        # Duplicate indices if requested batch_size > num_idx
        # Cast from tf.int32 to tf.float64 to ensure TF2.10-2.12 compatibility
        # with tf.math.divide_no_nan function
        reps = tf.math.ceil(tf.math.divide_no_nan(
                                            tf.cast(batch_size, tf.float64),
                                            tf.cast(idx.shape[0], tf.float64)))
        reps = tf.cast(tf.expand_dims(reps, axis=0), tf.int32)
        reps = tf.concat((reps, tf.ones_like(tf.cast(idx.shape[1:],tf.int32))),
                         axis=0)
        idx = tf.tile(idx, reps) # and repeat positions

        # Randomly permute indices
        idx = tf.random.shuffle(idx)

        # Sample batch_size random positions
        ue_pos = tf.gather_nd(self.cell_centers, idx[:batch_size])

        # Add random offset within cell-size, if positions should not be
        # centered
        if not center_pos:
            # cell can be rotated
            dir_x = tf.expand_dims(0.5*(cell_centers[0,0] - cell_centers[1,0]),
                                   axis=0)
            dir_y = tf.expand_dims(0.5*(cell_centers[0,0] - cell_centers[0,1]),
                                   axis=0)

            rand_x = tf.random.uniform((batch_size,1),
                                        minval=-1.,
                                        maxval=1.,
                                        dtype=self._rdtype)
            rand_y = tf.random.uniform((batch_size,1),
                                        minval=-1.,
                                        maxval=1.,
                                        dtype=self._rdtype)

            ue_pos += rand_x * dir_x + rand_y * dir_y

        return ue_pos

    def get_cell_index(self, pos):
        r"""get_cell_index(pos)

        Get the cell index of a position in the coverage map

        Input
        -----
        pos : [3], tf.float
            Position :math:`(x,y,z)` [m] in the global coordinate system

        Output
        ------
        : [2], tf.int
            Cell index :math:`(i,j)` of the position in the coverage map
        """

        if (tf.rank(pos) != 1) or (tf.shape(pos)[0] != 3):
            msg = "`pos` must be shaped as [x,y,z] (rank=1 and shape=[3])"
            raise ValueError(msg)

        pos = tf.cast(pos, self._rdtype)

        # Move to local coordinate system
        pos = pos - self._center
        rot_gcs_2_cm = tf.transpose(rotation_matrix(self._orientation))
        pos = tf.linalg.matvec(rot_gcs_2_cm, pos)

        # Get cell index
        pos = pos[:2] + self._size*0.5
        pos = tf.cast(tf.math.floor(pos/self._cell_size), tf.int32)

        return pos
    
    def get_value(self, pos):
        r"""get_value(pos)

        Get the value of the coverage map at a position

        Input
        -----
        pos : [3], tf.float
            Position :math:`(x,y,z)` [m] in the global coordinate system

        Output
        ------
        : tf.float
            Value of the coverage map at the position (linear scale)
        """

        if (tf.rank(pos) != 1) or (tf.shape(pos)[0] != 3):
            msg = "`pos` must be shaped as [x,y,z] (rank=1 and shape=[3])"
            raise ValueError(msg)

        pos = tf.cast(pos, self._rdtype)

        # Get cell index
        pos = self.get_cell_index(pos)
        # print("Cell index: ", pos)


        return self._value[:, pos[1], pos[0]]

    
    def to_mongo(self, collection, name="coverage_map"):
        r"""to_mongo(collection, name)
        Serializes the coverage map and related data to a MongoDB collection.

        Parameters
        ----------
        collection : pymongo.collection.Collection
            The MongoDB collection where the data will be stored.
        name : str
            The name of the dataset or configuration being saved.
        """
        # Ensure the MongoDB collection is provided
        if not isinstance(collection, pymongo.collection.Collection):
            raise ValueError("Invalid MongoDB collection provided.")

        # convert the tensor data to binary for storage
        data = self.as_tensor()
        data_binary = pickle.dumps(data)
        # Serialize data
        post = {
            "_id": name,
            "data": data_binary,
            # "center": self._center,  # Assuming TensorFlow tensors
            # "orientation": self._orientation.numpy().tolist(),
            # "size": self._size.numpy().tolist(),
            # "cell_size": self._cell_size,
            # "rdtype": str(self._rdtype),
            # "tx_positions": [pos.numpy().tolist() for pos in self._tx_pos],
            # "tx_orientations": [orient.numpy().tolist() for orient in self._tx_orientation],
            # "rx_positions": [pos.numpy().tolist() for pos in self._rx_pos],
            # "ris_positions": [pos.numpy().tolist() for pos in self._ris_pos],
            "center": pickle.dumps(self._center),
            "orientation": pickle.dumps(self._orientation),
            "size": pickle.dumps(self._size),
            "cell_size": pickle.dumps(self._cell_size),
            "tx_number": pickle.dumps(self.num_tx),
            "transmitters": list(self._transmitters.keys()),
            "tx_positions": pickle.dumps(self._tx_pos),
            "tx_orientations": pickle.dumps(self._tx_orientation),
            "rx_positions": pickle.dumps(self._rx_pos),
            "ris_positions": pickle.dumps(self._ris_pos),
            "rdtype": str(self._rdtype),
            # Include any other necessary data
        }
        # Insert data into MongoDB (overwrite if already exists)
        collection.update_one({"_id": name}, {"$set": post}, upsert=True)
        # Insert data into MongoDB
        # collection.insert_one(data)
        print(f"Coverage map data saved to MongoDB collection '{collection.name}'")
        
    def to_world(self):
        r"""
        Returns the `to_world` transformation that maps a default Mitsuba
        rectangle to the rectangle that defines the coverage map surface

        Output
        -------
        to_world : :class:`mitsuba.ScalarTransform4f`
            Rectangle to world transformation
        """
        return mitsuba_rectangle_to_world(self._center, self._orientation,
                                          self._size)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._tx_name_2_ind:
                raise ValueError(f"Unknown transmitter with name '{key}'")
            key = self._tx_name_2_ind[key]

        elif isinstance(key, (tuple, list)) and len(key) > 0:
            tx = key[0]

            if isinstance(tx, int):
                if tx >= self.num_tx:
                    raise ValueError("Invalid transmitter index:"\
                                    f" expected [0..{self.num_tx}], found {tx}")
            elif isinstance(tx, str):
                if tx not in self._tx_name_2_ind:
                    raise ValueError(f"Unknown transmitter with name '{tx}'")
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError("Invalid type for `tx`:"\
                                 " Must be a string or int")

            key = type(key)((
                tx, *key[1:]
            ))

        return self._value[key]
