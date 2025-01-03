# Interatomic contact matrices tools

## make_mat.py

This script calculates probabilities from histograms based on specified parameters.

Usage:
```
python make_mat.py --histo <histogram_directory> --target_top <target_topology> --mego_top <mego_topology> [--inter] [--out <output_path>] [--out_name <output_name>] [--proc <num_processes>] [--cutoff <max_cutoff>]
```
Parameters:

`--histo`: Path to the directory containing the histograms.
         Histogram files should contain the prefix "intra_" for intramolecular contact descriptions and "inter_" for intermolecular.
    
`--target_top`: Path to the topology file of the system on which the histograms were calculated.
    
`--mego_top`: Path to the standard multi-eGO topology of the system generated by pdb2gmx.
    
`--mode`: modality of the histogram analysis. Can be intra,same or cross (intramolecular, itermolecular same and intermolecular cross) or any combination by "+", e.g --mode intra+cross, --mode same+cross. If not provided all of them will be calculated --mode intra+same+cross.
    
-`-out`: Optional parameter to set the output path. Default is the current directory.
    
`--out_name`: Optional parameter to set the output name of files. It will be added to the default one.
                Example: intermat_<out_name>_mi_mj.ndx or intramat_<out_name>_mi_mj.ndx
    
`--num_threads`: Optional parameter to set the number of processes to perform the calculation. Default is 1.
    
`--cutoff`: The maximum cutoff used for the accumulation of the histograms.

`--zero`: Optional flag returning 0 distances and 0 probability matrices with the correct cutoff values and indeces.

## ndx2HDF5.py

This scripts convert a text or a text.gz matrix in HDF5 format, this is usefull for large system because it is much faster to process.

## HDF52ndx.py 

This scripts convert a HDF5 matrix into text file.
