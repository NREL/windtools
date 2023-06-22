import os
import numpy as np
import windtools.amrwind.post_processing as pp
import multiprocessing as mp

def to_ff(file_path,output_path,group_name,i_start,i_end,numcores,vtkstartind=0):
    '''
    An example function to write VTK files parallelly from AMR-Wind simulation.
    This code swapns a separate process to write each VTK file.
    The total number of parallel 
    
    Parameters:- 
    file_path: Path of the .nc file to be read
    output_path: Directory containing the groupwise folder where VTK files should be written
    group_name: Name of the group to be converted to VTK
    i_start: Initial index of the VTK file to be written (This is usually )
    i_end: Final index of the VTK file to be written
    numcores: Number of parallel processes spawned at a time. This should be set to 
        the number of cores in the CPU,
    vtkstartind: Index by which the names of the vtk files will be shifted. 
        This is useful for saving files starting from a non-zero time-step when AMR-Wind crashes
        unxpectedly and is restarted using a savefile.
    '''

    sample = pp.Sampling(file_path)
    print(sample)
    group_path = os.path.join(output_path,group_name)

    #     
    numbatches =  (i_end-i_start+1)//numcores+1
    for batchi in np.arange(numbatches):
        Proc_all = []
        for corei in np.arange(numcores):
            ind = batchi*numcores + corei + i_start
            print(ind)
            if ind>i_end:
                break
            p = mp.Process(target=parfunc,args=(sample,group_name,group_path,ind,ind+1,vtkstartind))
            p.start()
            Proc_all.append(p)
        for p in Proc_all:
            p.join()
            
def parfunc(sample,dataset,group_path,ind_i,ind_f,vtkstartind):
    #Function to be called by each script
    sample.to_vtk(dataset,group_path,offsetz=0, itime_i=ind_i,itime_f=ind_f,vtkstartind=vtkstartind)
