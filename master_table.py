
import numpy as np
from scipy import stats
from scipy.spatial import KDTree
from astropy.table import Table
from astropy.io import fits
import shutil
import os
from kai.reduce import calibrate
import seaborn as sns


import sys 
# align_copy_path='/g/ghez/abhimat/projects/2023_07_stf_migration/align_copy_files/'
sys.path.append("/Users/gcgstudent1/starfinder_version_compare")

import os
 
# change the current working directory
# to specified path
os.chdir('/Users/gcgstudent1/Sparsh')

from stf_runs_compare.file_readers import stf_lis_reader, stf_rms_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_pos_err_reader,\
    align_mag_reader, align_param_reader, align_name_reader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
import pandas as pd
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# epochs_quality_table = pd.read_table('/g/ghez/data/dr/dr1/data_quality/combo_epochs_quality/combo_epochs_quality_table_kp.txt',sep='\s+',skiprows=[1])

# for i in range(len(epochs_quality_table['epoch'])): 
#     epoch_name = epochs_quality_table.loc[i,'epoch']
#     print('Analysing ' + epoch_name)

#     try:
        
#     except:
#         pass

epoch_name = '20130426_0427nirc2'

## Important to add for cumulative analysis
epochs_quality_table = pd.read_table('/g/ghez/data/dr/dr1/data_quality/combo_epochs_quality/combo_epochs_quality_table_kp.txt',sep='\s+',skiprows=[1])

# master_table_stats = pd.DataFrame()

class compare_epoch(object):
    """
    Object to perform comparison between two different StarFinder runs,
    run on a single image or combined image
    """
    master_table_stats = pd.DataFrame()

    plate_scale = 0.00993   ## NIRC2 Plate Scale
    
    filt_label_strs = {
        'h': r"$H$-band",
        'kp': r"$K'$-band",
        'lp': r"$L'$-band",
    }
    
    filt_mag_label_strs = {
        'h': r"$m_{H}$",
        'kp': r"$m_{K'}$",
        'lp': r"$m_{L'}$",
    }
    
    filt_bright_mag_cutoff = {
        'h': 17,
        'kp': 15,
        'lp': 13,
    }
    
    def __init__(self,
        epoch_name,
        filt='kp',
        output_location='./',
        runs_in_dr=True,
        run1_dr_location='/g/ghez/data/dr/dr1/',
        run2_dr_location='/g/ghez/data/dr/dr1/',
        run1_dr_stf_version='v2_3',
        run2_dr_stf_version='v3_1',
        run1_location='',
        run2_location='',
        run1_stf_corr = '0.8',
        run2_stf_corr = '0.8',
        run1_file_name_str = 'run1',
        run2_file_name_str = 'run2',
        run1_name_str = 'Legacy',
        run2_name_str = 'Single-PSF',
    ):
        """
        Initializer function for compare_epoch object
        
        Parameters
        ----------
        epoch_name : str
            Name of the epoch, e.g.: 20220815nirc2
        filt : str, default='kp'
            String used for the observation filter in the filename
        output_location : str, default='./'
            String to indicate the output file path of all output files
        runs_in_dr : bool, default=True
            Boolean keyword to indicate whether or not the StarFinder
            runs to be compared are located in a GC imaging data release
            in the data release format.
        run1_dr_location : str, default='/g/ghez/data/dr/dr1/'
            Filepath of run 1's data release location
        run2_dr_location : str, default='/g/ghez/data/dr/dr1/'
            Filepath of run 2's data release location 
        run1_dr_stf_version : str, default='v2_3',
        run2_dr_stf_version : str, default='v3_1',
        run1_location : str, default='',
        run2_location : str, default='',
        run1_stf_corr : float, default='0.8',
        run2_stf_corr : float, default='0.8',
        run1_file_name_str : str, default='run1',
            Name of run 1, used in output files and table columns
        run2_file_name_str : str, default='run2',
            Name of run 2, used in output files and table columns
        run1_name_str : str, default='Legacy',
            Name of run 1, used in plots
        run2_name_str : str, default='Single-PSF',
            Name of run 2, used in plots
        """
        
        self.epoch_name = epoch_name
        self.epoch_filt = filt
        self.output_location = output_location
        
        # Store out variable names
        self.runs_in_dr = runs_in_dr
        self.run1_dr_location = run1_dr_location
        self.run2_dr_location = run2_dr_location
        self.run1_dr_stf_version = run1_dr_stf_version
        self.run2_dr_stf_version = run2_dr_stf_version
        
        self.run1_stf_corr = run1_stf_corr
        self.run2_stf_corr = run2_stf_corr
        
        self.run1_file_name_str = run1_file_name_str
        self.run2_file_name_str = run2_file_name_str
        self.run1_name_str = run1_name_str
        self.run2_name_str = run2_name_str
        self.correlation_compare_table = pd.DataFrame
        
       # QUESTION 1: 
       # What does " and runs_in_dr: " mean ?
       
        # Set up locations of StarFinder runs
        if run1_location == '' and runs_in_dr: 
            run1_location = f'{run1_dr_location}/starlists/combo/{self.epoch_name}/starfinder_{self.run1_dr_stf_version}/'
        
        if run2_location == '' and runs_in_dr:
            run2_location = f'{run2_dr_location}/starlists/combo/{self.epoch_name}/starfinder_{self.run2_dr_stf_version}/'
        
        self.run1_location = run1_location
        self.run2_location = run2_location
        
        # COMMENT: SUPER COOL WAY TO GET PATHS, {0},{1} are acting like placeholders.. 
        
        # Set up locations of align runs for the comparisons
        self.epoch_analysis_location = '{0}/{1}_{2}/'.format(
            self.output_location, self.epoch_name, self.epoch_filt,
        )
        
        self.align_dir = '{0}/{1}_{2}/starlists_align/'.format(
            self.output_location, self.epoch_name, self.epoch_filt,
        )
        
        self.align_rms_dir = '{0}/{1}_{2}/starlists_align_rms/'.format(
            self.output_location, self.epoch_name, self.epoch_filt,
        )
        
        return

    ## assuming there exits a grand master table which generate_master_table can access along 
    # with the data quality table. 

    def generate_master_table_stats(self):
        
        ## directory to read table 
        
        stats_table_dir = self.epoch_analysis_location + 'align_rms_stats/' + 'overall_stats.txt'
        stats_table = pd.read_table(stats_table_dir, sep= '\s+', skiprows= [1])
        
        sub_master_table_stats = pd.DataFrame()
        
        ## Copying data quality table values to the sub_master_table_stats
            # Bolean masking the epochs quality table to only have the current epochs info
            
        epochs_quality_table_mask= epochs_quality_table[epochs_quality_table['epoch'] == epoch_name]
        epochs_quality_table_mask= epochs_quality_table_mask.reset_index(drop=True)
        # print(epochs_quality_table_mask)
    
        for column in epochs_quality_table_mask.columns: 
        
            if column == 'single_multi' or column == 'MJD' or column == 'JYear': 
                pass 
            else: 
                sub_master_table_stats.loc[0,f'{column}'] = epochs_quality_table_mask.loc[0,column]
        
        ## copying values from stats table to sub_master_table_stats
        
        for column in stats_table.columns: 
            
            if column == 'run_file_name' or column == 'run_name': 
                pass 
            else:
                sub_master_table_stats.loc[0,'legacy_' + f'{column}'] = stats_table.loc[0,column]
                sub_master_table_stats.loc[0,'single_' + f'{column}'] = stats_table.loc[1,column]
    
        pd.options.display.float_format = '{:.3f}'.format
        
        # print(sub_master_table_stats)
        compare_epoch.master_table_stats = pd.concat([compare_epoch.master_table_stats, sub_master_table_stats], axis=0, ignore_index=True )

    

    ## Read the info in the table, make a new dataframe.. 



compare_epoch_object = compare_epoch(
            epoch_name,
            filt='kp',
            output_location='./',
            runs_in_dr=True,
            run1_dr_location='/g/ghez/data/dr/dr1/',
            run2_dr_location='/g/ghez/data/dr/dr1/',
            run1_dr_stf_version='v2_3',
            run2_dr_stf_version='v3_1',
            run1_stf_corr = '0.8',
            run2_stf_corr = '0.8',
            run1_file_name_str = 'run1',
            run2_file_name_str = 'run2',
            run1_name_str = 'Legacy',
            run2_name_str = 'Single-PSF',
        )


compare_epoch_object.generate_master_table_stats()

# compare_epoch_object1 = compare_epoch('20220514_0515nirc2')

# compare_epoch_object1.generate_master_table_stats()
# out_dir = './' + 'master_table'

# os.makedirs(out_dir, exist_ok=True)

compare_epoch.master_table_stats.to_csv('/Users/gcgstudent1/Sparsh/master_table.csv')
compare_epoch.master_table_stats.to_csv('/Users/gcgstudent1/Sparsh/master_table.txt',sep='\t')


