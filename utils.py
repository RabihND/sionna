import numpy as np
import matplotlib.pyplot as plt
import warnings
import pymongo
import pickle
import plotly.graph_objects as go

import tensorflow as tf


def get_value(pos, value):
    """ 
    Get the value of the coverage map at a specific position.

    Parameters
    ----------
    pos : tf.Tensor
        The position at which to get the value.
    value : tf.Tensor
        The coverage map data.
    
    Returns
    -------
    tf.Tensor
        The value of the coverage map at the specified position.
    """
    # pos tf.Tensor([107 164], shape=(2,), dtype=int32)
    # check if pos is a 1D tensor of shape (2,) or a list of length 2
    # if pos.shape  or pos.shape[0] != 2:
    #     raise ValueError("Invalid position provided. Expected a 1D tensor of shape (2,).")
    
    pos = tf.squeeze(pos)

    return value[:, pos[1], pos[0]]



def from_mongo(collection, name="coverage_map"):
    """
    Retrieves and unpickles the coverage map and related data from a MongoDB collection.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The MongoDB collection from which the data will be retrieved.
    name : str
        The name of the dataset or configuration being retrieved.

    Returns
    -------
    A dictionary containing the unpickled coverage map and related data.
    """
    # Ensure the MongoDB collection is provided
    if not isinstance(collection, pymongo.collection.Collection):
        raise ValueError("Invalid MongoDB collection provided.")

    # Fetch the document from MongoDB
    document = collection.find_one({"_id": name})
    if not document:
        raise ValueError(f"No document found with _id: {name}")

    # Unpickle the data
    unpickled_data = {
        "data": pickle.loads(document["data"]),
        "center": pickle.loads(document["center"]),
        "orientation": pickle.loads(document["orientation"]),
        "size": pickle.loads(document["size"]),
        "cell_size": pickle.loads(document["cell_size"]),
        "tx_number": pickle.loads(document["tx_number"]),
        "transmitters": document["transmitters"],
        "tx_positions": pickle.loads(document["tx_positions"]),
        "tx_orientations": pickle.loads(document["tx_orientations"]),
        "rx_positions": pickle.loads(document["rx_positions"]),
        "ris_positions": pickle.loads(document["ris_positions"]),
        "rdtype": document["rdtype"],
        # Include any other necessary data
    }

    return unpickled_data

def show(value, tx=0, vmin=None, vmax=None,
              show_tx=True,show_tx_orientation=False, show_rx=False,
              rx_positions=None,num_tx=None,tx_positions=None,tx_orientations=None,transmitters=None ):
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
        # print(transmitters)
        tx_name_2_ind = [i for i, tx in enumerate(transmitters)]
        # print(tx_name_2_ind)

        if isinstance(tx, int):
            if tx >= num_tx:
                raise ValueError("Invalid transmitter index")
        elif isinstance(tx, str):
            if tx in tx_name_2_ind:
                tx = tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        else:
            raise ValueError("Invalid type for `tx`: Must be a string or int")

        # Catch expected div-by-zero warnings
        with warnings.catch_warnings(record=True) as _:
            cm = 10.*np.log10(value[tx].numpy())

        # Position of the transmitter
        print(cm.shape)
        # Visualization the coverage map
        fig = plt.figure()
        plt.imshow(cm, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Path gain [dB]')
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        # Visualizing transmitter, receiver, RIS positions
        if show_tx:
            tx_pos = tx_positions[tx]
            fig.axes[0].scatter(*tx_pos, marker='P', c='r')

            if show_tx_orientation:
                # tx_orientation tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32)
                tx_orientation = tx_orientations[tx]
                # Calculate the direction vector for the arrow tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32) in rad
                dir_x = np.cos(tx_orientation[0])*np.cos(tx_orientation[1])
                dir_y = np.sin(tx_orientation[0])*np.cos(tx_orientation[1])
                fig.axes[0].arrow(tx_pos[0], tx_pos[1], dir_x, dir_y, head_width=0.1, head_length=0.1, fc='r', ec='r')
                

        if show_rx:
            for rx_pos in rx_positions:
                fig.axes[0].scatter(*rx_pos, marker='x', c='b')


        return fig


def show_plotly(value, tx=0, vmin=None, vmax=None,
                show_tx=True, show_tx_orientation=False, show_rx=False,
                rx_positions=None, num_tx=None, tx_positions=None, tx_orientations=None, transmitters=None):
    """
    Visualizes a coverage map using Plotly.

    Parameters
    ----------
    tx : int | str
        Index or name of the transmitter for which to show the coverage map. Defaults to 0.
    vmin, vmax : float | `None`
        Define the range of path gains that the colormap covers. If set to `None`, then covers the complete range. Defaults to `None`.
    show_tx : bool
        If set to `True`, then the position of the transmitter is shown. Defaults to `True`.
    show_tx_orientation : bool
        If set to `True`, then the orientation of the transmitter is shown. Defaults to `False`.
    show_rx : bool
        If set to `True`, then the position of the receivers is shown. Defaults to `False`.
    rx_positions : list
        List of receiver positions.
    num_tx : int
        Number of transmitters.
    tx_positions : list
        List of transmitter positions.
    tx_orientations : list
        List of transmitter orientations.
    transmitters : list
        List of transmitters.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figure showing the coverage map.
    """
    if isinstance(tx, str):
        tx = transmitters.index(tx) if tx in transmitters else None
        if tx is None:
            raise ValueError(f"Unknown transmitter with name '{tx}'")
    elif isinstance(tx, int):
        if tx >= num_tx:
            raise ValueError("Invalid transmitter index")
    else:
        raise ValueError("Invalid type for `tx`: Must be a string or int")

    # Catch expected div-by-zero warnings
    with warnings.catch_warnings(record=True) as _:
        cm = 10.*np.log10(value[tx].numpy())

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax
    ))

    fig.update_layout(
        title="Coverage Map",
        xaxis_title="Cell index (X-axis)",
        yaxis_title="Cell index (Y-axis)"
    )

    if show_tx:
        tx_pos = tx_positions[tx]
        fig.add_trace(go.Scatter(x=[tx_pos[0]], y=[tx_pos[1]], mode='markers', marker=dict(color='red', symbol='x')))

        if show_tx_orientation:
            tx_orientation = tx_orientations[tx]
            dir_x = np.cos(tx_orientation[0]) * np.cos(tx_orientation[1])
            dir_y = np.sin(tx_orientation[0]) * np.cos(tx_orientation[1])
            fig.add_trace(go.Scatter(x=[tx_pos[0], tx_pos[0] + dir_x], y=[tx_pos[1], tx_pos[1] + dir_y], mode='lines', line=dict(color='red')))

    if show_rx:
        for rx_pos in rx_positions:
            fig.add_trace(go.Scatter(x=[rx_pos[0]], y=[rx_pos[1]], mode='markers', marker=dict(color='blue', symbol='circle')))

    return fig


def show_sinr(value, tx=0, vmin=None, vmax=None,
              show_tx=True,show_tx_orientation=False, show_rx=False,
              rx_positions=None,num_tx=None,tx_positions=None,tx_orientations=None,transmitters=None ):
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
    # print(transmitters)
    tx_name_2_ind = [i for i, tx in enumerate(transmitters)]
    # print(tx_name_2_ind)

    if isinstance(tx, int):
        if tx >= num_tx:
            raise ValueError("Invalid transmitter index")
    elif isinstance(tx, str):
        if tx in tx_name_2_ind:
            tx = tx_name_2_ind[tx]
        else:
            raise ValueError(f"Unknown transmitter with name '{tx}'")
    else:
        raise ValueError("Invalid type for `tx`: Must be a string or int")

    # Catch expected div-by-zero warnings
    with warnings.catch_warnings(record=True) as _:
        cm = 10.*np.log10(value[tx].numpy())
        # total interference equal the value of the coverage map of all transmitters except the current transmitter
        sinr = tf.zeros_like(cm)
        for i in range(num_tx):
            if i != tx:
                sinr += 10.*np.log10(value[i].numpy())
        sinr = tf.subtract(cm, sinr)
        # sinr = np.rot90(sinr, k=1)
            
    # Position of the transmitter
    # print(f"SNIR at the receiver position: {sinr}")
    # Visualization the coverage map
    fig = plt.figure()
    plt.imshow(sinr, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='SINR [dB]')
    plt.xlabel('Cell index (X-axis)')
    plt.ylabel('Cell index (Y-axis)')
    # Visualizing transmitter, receiver, RIS positions
    if show_tx:
        tx_pos = tx_positions[tx]
        fig.axes[0].scatter(*tx_pos, marker='P', c='r')

        if show_tx_orientation:
            # tx_orientation tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32)
            tx_orientation = tx_orientations[tx]
            # Calculate the direction vector for the arrow tf.Tensor([ 1.3114178  -0.01978755  0.        ], shape=(3,), dtype=float32) in rad
            dir_x = np.cos(tx_orientation[0])*np.cos(tx_orientation[1])
            dir_y = np.sin(tx_orientation[0])*np.cos(tx_orientation[1])
            fig.axes[0].arrow(tx_pos[0], tx_pos[1], dir_x, dir_y, head_width=0.1, head_length=0.1, fc='r', ec='r')
            

    if show_rx:
        for rx_pos in rx_positions:
            fig.axes[0].scatter(*rx_pos, marker='x', c='b')


    return fig