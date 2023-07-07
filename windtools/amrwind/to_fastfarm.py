import os
import numpy as np
import multiprocessing as mp
from mpi4py import MPI

import windtools.amrwind.post_processing as pp


def dristribute_nodes(
    file_path, output_path, group_name, i_start, i_end, vtkstartind=0
):
    """
    An example function to write VTK files parallelly from AMR-Wind simulation using MPI

    Args:
    -----
    file_path: str
        Path of the .nc file to be read
    output_path: str
        Directory containing the groupwise folder where VTK files should be written
    group_name: str
        Name of the group to be converted to VTK
    i_start: int
        Initial index of the VTK file to be written (This is usually )
    i_end: int
        Final index of the VTK file to be written
    vtkstartind: int, optional, default=0
        Index by which the names of the vtk files will be shifted.
        This is useful for saving files starting from a non-zero time-step when AMR-Wind crashes
        unxpectedly and is restarted using a savefile.

    Example:
    -------
    The function can be used as follows:

    >>> import windtools.amrwind.to_fastfarm as tf
    >>> import time
    >>>
    >>> file_path = '../sampling_lr96000.nc'
    >>> output_path = 'output'
    >>> group_name = 'Low'
    >>> numcores = 36
    >>> shiftind=0
    >>> i_start = 0
    >>> i_end = 7500
    >>>
    >>> startTime = time.time()
    >>> tf.distribute_nodes(file_path,output_path,group_name,i_start,i_end,shiftind)
    >>> executionTime = (time.time() - startTime)
    >>> print('Execution time in seconds: ' + str(executionTime))

    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate the workload for each process
    workpernode = (i_end - i_start + 1) // size
    remainder = (i_end - i_start + 1) % size

    # Determine the start and end indices for each process
    start_index = i_start + rank * workpernode + min(rank, remainder)
    end_index = start_index + workpernode + (rank < remainder)
    print(f"The current rank is {rank}")
    disritube_cores(
        file_path, output_path, group_name, start_index, end_index, vtkstartind=0
    )


def disritube_cores(file_path, output_path, group_name, i_start, i_end, vtkstartind=0):

    # Find number of cores
    numcores = mp.cpu_count()
    # numcores = 2

    # Read sample and prints the groups contained in it
    sample = pp.Sampling(file_path)
    print(sample)
    group_path = os.path.join(output_path, group_name)
    # dst1 = sample.read_single_group('Low')

    # numbatches is the number of batches of `numcores` processes that will be spawned
    numbatches = (i_end - i_start + 1) // numcores + 1

    for batchi in np.arange(numbatches):
        Proc_all = []  # List to stores all the processes for a batch.
        for corei in np.arange(numcores):
            ind = (
                batchi * numcores + corei + i_start
            )  # Index of the VTK file to be written
            if ind > i_end:  # Don't spawn more processes if last index is reached
                break

            # Spawn one process for each index
            p = mp.Process(
                target=parfunc,
                args=(sample, group_name, group_path, ind, ind + 1, vtkstartind),
            )

            # Start the process
            p.start()

            # Store all the processes spawned in a batch in a list
            Proc_all.append(p)

        # Only move the the next batch after all the processes of the current batch are completed
        for p in Proc_all:
            p.join()


def parfunc(sample, dataset, group_path, ind_i, ind_f, vtkstartind):
    # Function to be called by each process spawned
    sample.to_vtk(
        dataset,
        group_path,
        offsetz=0,
        itime_i=ind_i,
        itime_f=ind_f,
        vtkstartind=vtkstartind,
        verbose=False,
    )
