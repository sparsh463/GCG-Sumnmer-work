# Class to perform comparison between two starfinder versions
# ---
# Abhimat Gautam

import numpy as np
from scipy import stats
from scipy.spatial import KDTree
from astropy.table import Table
from astropy.io import fits
import shutil
import os
from kai.reduce import calibrate
from stf_runs_compare.file_readers import stf_lis_reader, stf_rms_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_pos_err_reader,\
    align_mag_reader, align_param_reader, align_name_reader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
import pandas as pd
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


class compare_epoch(object):
    """
    Object to perform comparison between two different StarFinder runs,
    run on a single image or combined image
    """
    
    plate_scale = 0.00993   ## NIRC2 Plate Scale
    
    master_table_stats = pd.DataFrame()
    epochs_quality_table = pd.read_table('/g/ghez/data/dr/dr1/data_quality/combo_epochs_quality/combo_epochs_quality_table_kp.txt',sep='\s+',skiprows=[1])

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
    
    
    def run_align(self,
        align_flags='-r align_d -m 16 -a 0 -p -v -N ../source_list/label_abs.dat -o ../source_list/orbits.dat -O 2 align_d.list',
        align_abs_flags='-a 3 -abs ../source_list/absolute_refs.dat align_d align_d_abs ',
        align_copy_path='./align_copy_files/', 
        cal_stars = 'irs16NW,S3-22,S1-17,S1-34,S4-3,S1-1,S1-21,S3-370,S3-88,S3-36,S2-63',
        cal_align_stars = 'irs16C,irs16NW,irs16CC',
        cal_first_star = 'irs16C',
    ):
        """
        Run align on the two different starfinder runs for the current epoch 
        """
        # Get original working directory, to change back into after the different steps
        orig_wd = os.getcwd()
        
        # Create directory for the align run between the two modes
        cur_epoch_align_dir = self.align_dir
        os.makedirs(cur_epoch_align_dir, exist_ok=True)
        
        # Remove existing, and copy source_list directory
        if os.path.exists('{0}/source_list'.format(cur_epoch_align_dir)):
            shutil.rmtree(
                '{0}/source_list'.format(cur_epoch_align_dir),
                ignore_errors=True,
            )
        
        shutil.copytree(
            '{0}/source_list'.format(align_copy_path),
            '{0}/source_list'.format(cur_epoch_align_dir),
            dirs_exist_ok=True,
        )
        
        # Make lis directory for align run and copy starlists
        lis_dir_loc = '{0}/lis/'.format(cur_epoch_align_dir)
        
        if os.path.exists(lis_dir_loc): # Remove any existing lis directories
            shutil.rmtree(
                lis_dir_loc,
                ignore_errors=True,
            )
        
        os.makedirs(lis_dir_loc, exist_ok=True)
        
        # Original starlist names
        orig_run1_stf_file = '{0}/mag{1}_{2}_{3}_stf.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt, self.run1_stf_corr,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_{3}_stf.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt, self.run2_stf_corr,
        )
        
        # Copied starlist names into align
        # (need to make sure the names are unique)
        new_run1_stf_file = 'mag{0}_{1}_{2}_stf_{3}.lis'.format(
            self.epoch_name, self.epoch_filt, self.run1_stf_corr,
            self.run1_file_name_str,
        )
        new_run2_stf_file = 'mag{0}_{1}_{2}_stf_{3}.lis'.format(
            self.epoch_name, self.epoch_filt, self.run2_stf_corr,
            self.run2_file_name_str,
        )
        
        shutil.copy(orig_run1_stf_file, lis_dir_loc + new_run1_stf_file)
        shutil.copy(orig_run2_stf_file, lis_dir_loc + new_run2_stf_file)
        
        # Run calibrate on both copied starlists
        os.chdir(lis_dir_loc)
        
        self.run_calibrate(
            new_run1_stf_file,
            cal_stars=cal_stars,
            align_stars=cal_align_stars,
            cal_first_star=cal_first_star,
            align_rms_format=False,
        )
        self.run_calibrate(
            new_run2_stf_file,
            cal_stars=cal_stars,
            align_stars=cal_align_stars,
            cal_first_star=cal_first_star,
            align_rms_format=False,
        )
        
        os.chdir(orig_wd)
        
        # Make align directory
        align_dir_loc = '{0}/align/'.format(cur_epoch_align_dir)
        os.makedirs(align_dir_loc, exist_ok=True)
        
        # Make align_d.list file with the calibrated starlists
        cal_run1_stf_file = 'mag{0}_{1}_{2}_stf_{3}_cal.lis'.format(
            self.epoch_name, self.epoch_filt, self.run1_stf_corr,
            self.run1_file_name_str,
        )
        cal_run2_stf_file = 'mag{0}_{1}_{2}_stf_{3}_cal.lis'.format(
            self.epoch_name, self.epoch_filt, self.run2_stf_corr,
            self.run2_file_name_str,
        )
        
        list_file = ''
        list_file += '../lis/{0} 8 \n'.format(cal_run1_stf_file)
        list_file += '../lis/{0} 8 '.format(cal_run2_stf_file)
        
        with open(align_dir_loc + 'align_d.list', 'w') as out_file:
            out_file.write(list_file)
        
        # Run align
        os.chdir(align_dir_loc)
        
        java_align_command = 'java align ' + align_flags
        os.system(java_align_command)
        
        java_align_abs_command = 'java align_absolute ' + align_abs_flags
        os.system(java_align_abs_command)
        
        os.chdir(orig_wd)
    
    def run_align_rms(self,
        align_flags='-r align_d_rms -m 16 -a 0 -p -v -N ../source_list/label_abs.dat -o ../source_list/orbits.dat -O 2 align_d_rms.list',
        align_abs_flags='-a 3 -abs ../source_list/absolute_refs.dat align_d_rms align_d_rms_abs ',
        trim_align_flags = '-r align_d_rms_abs_t -e 1 -p -f ../points align_d_rms_abs',
        align_copy_path='./align_copy_files/', 
        cal_stars = 'irs16NW,S3-22,S1-17,S1-34,S4-3,S1-1,S1-21,S3-370,S3-88,S3-36,S2-63',
        cal_align_stars = 'irs16C,irs16NW,irs16CC',
        cal_first_star = 'irs16C',
    ):
        """
        Run align_rms on the two different starfinder runs for the current epoch 
        """
        # Get original working directory, to change back into after the different steps
        orig_wd = os.getcwd()
        
        # Create directory for the align run between the two modes
        cur_epoch_align_dir = self.align_rms_dir
        os.makedirs(cur_epoch_align_dir, exist_ok=True)
        
        # Remove existing, and copy source_list directory
        if os.path.exists('{0}/source_list'.format(cur_epoch_align_dir)):
            shutil.rmtree(
                '{0}/source_list'.format(cur_epoch_align_dir),
                ignore_errors=True,
            )
        
        shutil.copytree(
            '{0}/source_list'.format(align_copy_path),
            '{0}/source_list'.format(cur_epoch_align_dir),
            dirs_exist_ok=True,
        )
        
        # Make lis directory for align run and copy starlists
        lis_dir_loc = '{0}/lis/'.format(cur_epoch_align_dir)
        
        if os.path.exists(lis_dir_loc): # Remove any existing lis directories
            shutil.rmtree(
                lis_dir_loc,
                ignore_errors=True,
            )
        
        os.makedirs(lis_dir_loc, exist_ok=True)
        
        # Original starlist names
        orig_run1_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt,
        )
        
        # Copied starlist names into align
        # (need to make sure the names are unique)
        new_run1_stf_file = 'mag{0}_{1}_rms_{2}.lis'.format(
            self.epoch_name, self.epoch_filt,
            self.run1_file_name_str,
        )
        new_run2_stf_file = 'mag{0}_{1}_rms_{2}.lis'.format(
            self.epoch_name, self.epoch_filt,
            self.run2_file_name_str,
        )
        
        shutil.copy(orig_run1_stf_file, lis_dir_loc + new_run1_stf_file)
        shutil.copy(orig_run2_stf_file, lis_dir_loc + new_run2_stf_file)
        
        # Run calibrate on both copied starlists
        os.chdir(lis_dir_loc)
        
        self.run_calibrate(
            new_run1_stf_file,
            cal_stars=cal_stars,
            align_stars=cal_align_stars,
            cal_first_star=cal_first_star,
            align_rms_format=True,
        )
        self.run_calibrate(
            new_run2_stf_file,
            cal_stars=cal_stars,
            align_stars=cal_align_stars,
            cal_first_star=cal_first_star,
            align_rms_format=True,
        )
        
        os.chdir(orig_wd)
        
        # Make align directory
        align_dir_loc = '{0}/align/'.format(cur_epoch_align_dir)
        os.makedirs(align_dir_loc, exist_ok=True)
        
        # Make align_d.list file with the calibrated starlists
        cal_run1_stf_file = 'mag{0}_{1}_rms_{2}_cal.lis'.format(
            self.epoch_name, self.epoch_filt,
            self.run1_file_name_str,
        )
        cal_run2_stf_file = 'mag{0}_{1}_rms_{2}_cal.lis'.format(
            self.epoch_name, self.epoch_filt,
            self.run2_file_name_str,
        )
        
        list_file = ''
        list_file += '../lis/{0} 9 \n'.format(cal_run1_stf_file)
        list_file += '../lis/{0} 9 '.format(cal_run2_stf_file)
        
        with open(align_dir_loc + 'align_d_rms.list', 'w') as out_file:
            out_file.write(list_file)
        
        # Run align
        os.chdir(align_dir_loc)
        
        java_align_command = 'java align ' + align_flags
        os.system(java_align_command)
        
        java_align_abs_command = 'java align_absolute ' + align_abs_flags
        os.system(java_align_abs_command)
        
        # # Run trim align to create points directory
        # os.makedirs('../points', exist_ok=True)
        #
        # java_trim_align_command = 'java -Xmx2048m trim_align ' + trim_align_flags
        # os.system(java_trim_align_command)
        
        os.chdir(orig_wd)
    
    def run_calibrate(
        self,
        starlist,
        cal_stars = 'irs16NW,S3-22,S1-17,S1-34,S4-3,S1-1,S1-21,S3-370,S3-88,S3-36,S2-63',
        align_stars = 'irs16C,irs16NW,irs16CC',
        cal_first_star = 'irs16C',
        align_rms_format = False,
    ):
        """
        Run calibrate on a given starlist
        """
        
        calibrate_args = ''
        
        if align_rms_format:
            calibrate_args += '-f 2 '
        else:
            calibrate_args += '-f 1 '
        calibrate_args += '-R -V '
        calibrate_args += '-N ../source_list/photo_calib.dat '
        calibrate_args += '-M Kp -T 0 '
        
        calibrate_args += '-S {0} '.format(cal_stars)
        calibrate_args += '-A {0} '.format(align_stars)
        calibrate_args += '-I {0} '.format(cal_first_star)
        
        calibrate_args += '-c 4 '
        
        calibrate_args += starlist
        
        print('calibrate ' + calibrate_args)
        
        calibrate.main(calibrate_args.split())
    
    def construct_align_stf_errs_table(
        self,
        align_name_table,
        run1_stf_rms_table,
        run2_stf_rms_table,
    ):
        """
        Construct an error table with align star names using the pos and mag
        errors stored in the starfinder RMS file
        """
        
        run1_stf_x_errs = np.empty(len(align_name_table))
        run2_stf_x_errs = np.empty(len(align_name_table))
        
        run1_stf_y_errs = np.empty(len(align_name_table))
        run2_stf_y_errs = np.empty(len(align_name_table))
        
        run1_stf_m_errs = np.empty(len(align_name_table))
        run2_stf_m_errs = np.empty(len(align_name_table))
        
        for (row_index, row) in enumerate(align_name_table):
            name_run1 = row[self.run1_file_name_str + '_name']
            name_run2 = row[self.run2_file_name_str + '_name']
            
            if name_run1 != '-':
                run1_stf_x_errs[row_index] =\
                    (run1_stf_rms_table.loc[name_run1])['xe']
                run1_stf_y_errs[row_index] =\
                    (run1_stf_rms_table.loc[name_run1])['ye']
                run1_stf_m_errs[row_index] =\
                    (run1_stf_rms_table.loc[name_run1])['me']
            else:
                run1_stf_x_errs[row_index] = np.nan
                run1_stf_y_errs[row_index] = np.nan
                run1_stf_m_errs[row_index] = np.nan
            
            if name_run2 != '-':
                run2_stf_x_errs[row_index] =\
                    (run2_stf_rms_table.loc[name_run2])['xe']
                run2_stf_y_errs[row_index] =\
                    (run2_stf_rms_table.loc[name_run2])['ye']
                run2_stf_m_errs[row_index] =\
                    (run2_stf_rms_table.loc[name_run2])['me']
            else:
                run2_stf_x_errs[row_index] = np.nan
                run2_stf_y_errs[row_index] = np.nan
                run2_stf_m_errs[row_index] = np.nan
        
        align_stf_errs_table = Table(
            [
                align_name_table['name'],
                run1_stf_x_errs,
                run2_stf_x_errs,
                run1_stf_y_errs,
                run2_stf_y_errs,
                run1_stf_m_errs,
                run2_stf_m_errs,
            ],
            names=(
                'name',
                self.run1_file_name_str + '_xe',
                self.run2_file_name_str + '_xe',
                self.run1_file_name_str + '_ye',
                self.run2_file_name_str + '_ye',
                self.run1_file_name_str + '_me',
                self.run2_file_name_str + '_me',
            )
        )
        
        align_stf_errs_table.add_index('name')
        
        return align_stf_errs_table
    
    def compute_overall_stats_rms(
        self,
    ):
        cur_wd = os.getcwd()
        
        starlist_align_location = self.align_rms_dir + '/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'
        
        out_dir = self.epoch_analysis_location + 'align_rms_stats/'
        os.makedirs(out_dir, exist_ok=True)
        
        # Read in STF tables
        run1_stf_cal_file = '{0}/lis/mag{1}_{2}_rms_{3}_cal.lis'.format(
            self.align_rms_dir,
            self.epoch_name, self.epoch_filt,
            self.run1_file_name_str,
        )
        run2_stf_cal_file = '{0}/lis/mag{1}_{2}_rms_{3}_cal.lis'.format(
            self.align_rms_dir,
            self.epoch_name, self.epoch_filt,
            self.run2_file_name_str,
        )
        
        stf_run1_lis_table = stf_rms_lis_reader(run1_stf_cal_file)
        stf_run2_lis_table = stf_rms_lis_reader(run2_stf_cal_file)
        
        # Read in align tables
        align_mag_table = align_mag_reader(
            align_root + '.mag',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_pos_table = align_pos_reader(
            align_root + '.pos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_pos_err_table = align_pos_err_reader(
            align_root + '.err',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_orig_pos_table = align_orig_pos_reader(
            align_root + '.origpos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_param_table = align_param_reader(
            align_root + '.param',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_name_table = align_name_reader(
            align_root + '.name',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        # Construct a mag error table from stf rms lis table
        align_stf_errs_table = self.construct_align_stf_errs_table(
            align_name_table,
            stf_run1_lis_table,
            stf_run2_lis_table,
        )
        
        # QUESTION: HOW IS THIS FILTER WORKING? 
        
        # Common detections in each starlist
        det_both = np.logical_and(
            align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
            align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
        )
        
        detection_filter = np.where(det_both)
        
        common_detections_mag_table = align_mag_table[detection_filter]
        common_detections_pos_table = align_pos_table[detection_filter]
        common_detections_pos_err_table = align_pos_err_table[detection_filter]
        common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
        common_detections_param_table = align_param_table[detection_filter]
        common_detections_name_table = align_name_table[detection_filter]
        common_detections_stf_errs_table = align_stf_errs_table[detection_filter]
        
        common_detections_r2d_table = np.hypot(
            common_detections_pos_table[self.run1_file_name_str + '_x'],
            common_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        # One mode detections
        run1_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] == 0.0,
            ),
        )
        run2_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] == 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
            ),
        )
        
        run1_only_detections_mag_table = align_mag_table[run1_detection_filter]
        run1_only_detections_pos_table = align_pos_table[run1_detection_filter]
        run1_only_detections_pos_err_table = align_pos_err_table[run1_detection_filter]
        run1_only_detections_orig_pos_table = align_orig_pos_table[run1_detection_filter]
        run1_only_detections_param_table = align_param_table[run1_detection_filter]
        run1_only_detections_name_table = align_name_table[run1_detection_filter]
        run1_only_detections_stf_errs_table = align_stf_errs_table[run1_detection_filter]
        
        run1_only_detections_r2d_table = np.hypot(
            run1_only_detections_pos_table[self.run1_file_name_str + '_x'],
            run1_only_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        run2_only_detections_mag_table = align_mag_table[run2_detection_filter]
        run2_only_detections_pos_table = align_pos_table[run2_detection_filter]
        run2_only_detections_pos_err_table = align_pos_err_table[run2_detection_filter]
        run2_only_detections_orig_pos_table = align_orig_pos_table[run2_detection_filter]
        run2_only_detections_param_table = align_param_table[run2_detection_filter]
        run2_only_detections_name_table = align_name_table[run2_detection_filter]
        run2_only_detections_stf_errs_table = align_stf_errs_table[run2_detection_filter]
        
        run2_only_detections_r2d_table = np.hypot(
            run2_only_detections_pos_table[self.run2_file_name_str + '_x'],
            run2_only_detections_pos_table[self.run2_file_name_str + '_y'],
        )
        
        # Position differences
        diff_pos_x = (common_detections_orig_pos_table[self.run2_file_name_str + '_x'] -
                      common_detections_orig_pos_table[self.run1_file_name_str + '_x'])
        diff_pos_y = (common_detections_orig_pos_table[self.run2_file_name_str + '_y'] -
                      common_detections_orig_pos_table[self.run1_file_name_str + '_y'])
        
        common_x = common_detections_orig_pos_table[self.run1_file_name_str + '_x']
        common_y = common_detections_orig_pos_table[self.run1_file_name_str + '_y']
        
        if len(diff_pos_x) < 1:
            print('No stars in common between the starlists found')
            
            return
        
        # Number of detections
        run1_detection_filter = np.where(
            align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
        )
        run2_detection_filter = np.where(
            align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
        )
        
        run1_num_dets = len(align_mag_table[run1_detection_filter])
        run2_num_dets = len(align_mag_table[run2_detection_filter])
        
        # Number of bright detections
        run1_bright_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
                align_mag_table[self.run1_file_name_str + '_mag'] <\
                    self.filt_bright_mag_cutoff[self.epoch_filt],
            ),
        )
        run2_bright_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] <\
                    self.filt_bright_mag_cutoff[self.epoch_filt],
            ),
        )
        
        run1_num_bright_dets = len(align_mag_table[run1_bright_detection_filter])
        run2_num_bright_dets = len(align_mag_table[run2_bright_detection_filter])
        
        # STF Position and Mag errors
        run1_stf_x_errs = align_stf_errs_table[run1_detection_filter][self.run1_file_name_str + '_xe']
        run1_stf_bright_x_errs = align_stf_errs_table[run1_bright_detection_filter][self.run1_file_name_str + '_xe']
        
        run1_stf_y_errs = align_stf_errs_table[run1_detection_filter][self.run1_file_name_str + '_ye']
        run1_stf_bright_y_errs = align_stf_errs_table[run1_bright_detection_filter][self.run1_file_name_str + '_ye']
        
        run1_stf_m_errs = align_stf_errs_table[run1_detection_filter][self.run1_file_name_str + '_me']
        run1_stf_bright_m_errs = align_stf_errs_table[run1_bright_detection_filter][self.run1_file_name_str + '_me']
        
        run2_stf_x_errs = align_stf_errs_table[run2_detection_filter][self.run2_file_name_str + '_xe']
        run2_stf_bright_x_errs = align_stf_errs_table[run2_bright_detection_filter][self.run2_file_name_str + '_xe']
        
        run2_stf_y_errs = align_stf_errs_table[run2_detection_filter][self.run2_file_name_str + '_ye']
        run2_stf_bright_y_errs = align_stf_errs_table[run2_bright_detection_filter][self.run2_file_name_str + '_ye']
        
        run2_stf_m_errs = align_stf_errs_table[run2_detection_filter][self.run2_file_name_str + '_me']
        run2_stf_bright_m_errs = align_stf_errs_table[run2_bright_detection_filter][self.run2_file_name_str + '_me']
        
        
        run1_stf_x_errs_med = np.median(run1_stf_x_errs)
        run1_stf_bright_x_errs_med = np.median(run1_stf_bright_x_errs)
        run1_stf_y_errs_med = np.median(run1_stf_y_errs)
        run1_stf_bright_y_errs_med = np.median(run1_stf_bright_y_errs)
        run1_stf_m_errs_med = np.median(run1_stf_m_errs)
        run1_stf_bright_m_errs_med = np.median(run1_stf_bright_m_errs)
        run2_stf_x_errs_med = np.median(run2_stf_x_errs)
        run2_stf_bright_x_errs_med = np.median(run2_stf_bright_x_errs)
        run2_stf_y_errs_med = np.median(run2_stf_y_errs)
        run2_stf_bright_y_errs_med = np.median(run2_stf_bright_y_errs)
        run2_stf_m_errs_med = np.median(run2_stf_m_errs)
        run2_stf_bright_m_errs_med = np.median(run2_stf_bright_m_errs)
        
        run1_stf_x_errs_mad = stats.median_abs_deviation(run1_stf_x_errs)
        run1_stf_bright_x_errs_mad = stats.median_abs_deviation(run1_stf_bright_x_errs)
        run1_stf_y_errs_mad = stats.median_abs_deviation(run1_stf_y_errs)
        run1_stf_bright_y_errs_mad = stats.median_abs_deviation(run1_stf_bright_y_errs)
        run1_stf_m_errs_mad = stats.median_abs_deviation(run1_stf_m_errs)
        run1_stf_bright_m_errs_mad = stats.median_abs_deviation(run1_stf_bright_m_errs)
        run2_stf_x_errs_mad = stats.median_abs_deviation(run2_stf_x_errs)
        run2_stf_bright_x_errs_mad = stats.median_abs_deviation(run2_stf_bright_x_errs)
        run2_stf_y_errs_mad = stats.median_abs_deviation(run2_stf_y_errs)
        run2_stf_bright_y_errs_mad = stats.median_abs_deviation(run2_stf_bright_y_errs)
        run2_stf_m_errs_mad = stats.median_abs_deviation(run2_stf_m_errs)
        run2_stf_bright_m_errs_mad = stats.median_abs_deviation(run2_stf_bright_m_errs)
        
        # Align position errors
        run1_align_x_errs = align_pos_err_table[run1_detection_filter][self.run1_file_name_str + '_x_err']
        run1_align_bright_x_errs = align_pos_err_table[run1_bright_detection_filter][self.run1_file_name_str + '_x_err']
        
        run1_align_y_errs = align_pos_err_table[run1_detection_filter][self.run1_file_name_str + '_y_err']
        run1_align_bright_y_errs = align_pos_err_table[run1_bright_detection_filter][self.run1_file_name_str + '_y_err']
        
        run2_align_x_errs = align_pos_err_table[run2_detection_filter][self.run2_file_name_str + '_x_err']
        run2_align_bright_x_errs = align_pos_err_table[run2_bright_detection_filter][self.run2_file_name_str + '_x_err']
        
        run2_align_y_errs = align_pos_err_table[run2_detection_filter][self.run2_file_name_str + '_y_err']
        run2_align_bright_y_errs = align_pos_err_table[run2_bright_detection_filter][self.run2_file_name_str + '_y_err']
        
        
        run1_align_x_errs_med = np.median(run1_align_x_errs)
        run1_align_bright_x_errs_med = np.median(run1_align_bright_x_errs)
        run1_align_y_errs_med = np.median(run1_align_y_errs)
        run1_align_bright_y_errs_med = np.median(run1_align_bright_y_errs)
        
        run2_align_x_errs_med = np.median(run2_align_x_errs)
        run2_align_bright_x_errs_med = np.median(run2_align_bright_x_errs)
        run2_align_y_errs_med = np.median(run2_align_y_errs)
        run2_align_bright_y_errs_med = np.median(run2_align_bright_y_errs)
        
        run1_align_x_errs_mad = stats.median_abs_deviation(run1_align_x_errs)
        run1_align_bright_x_errs_mad = stats.median_abs_deviation(run1_align_bright_x_errs)
        run1_align_y_errs_mad = stats.median_abs_deviation(run1_align_y_errs)
        run1_align_bright_y_errs_mad = stats.median_abs_deviation(run1_align_bright_y_errs)
        
        run2_align_x_errs_mad = stats.median_abs_deviation(run2_align_x_errs)
        run2_align_bright_x_errs_mad = stats.median_abs_deviation(run2_align_bright_x_errs)
        run2_align_y_errs_mad = stats.median_abs_deviation(run2_align_y_errs)
        run2_align_bright_y_errs_mad = stats.median_abs_deviation(run2_align_bright_y_errs)
        
        ## Central arcsec detections analysis
        
        run1_only_detections_r2d_table = np.hypot(
            run1_only_detections_pos_table[self.run1_file_name_str + '_x'],
            run1_only_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        run2_only_detections_r2d_table = np.hypot(
            run2_only_detections_pos_table[self.run2_file_name_str + '_x'],
            run2_only_detections_pos_table[self.run2_file_name_str + '_y'],
        )
        
        central_arcsec_filt_run1 = np.where(run1_only_detections_r2d_table < 1)
        central_arcsec_filt_run2 = np.where(run2_only_detections_r2d_table < 1)
        
        
        ## Central num_detections
        run1_num_central_dets = len(run1_only_detections_r2d_table[central_arcsec_filt_run1])
        run2_num_central_dets = len(run2_only_detections_r2d_table[central_arcsec_filt_run2])
        
        ## Errors 
        
        run1_align_central_x_errs =  run1_align_x_errs[central_arcsec_filt_run1]
        run2_align_central_x_errs =  run2_align_x_errs[central_arcsec_filt_run2]
        
        run1_align_central_x_errs_med = np.median(run1_align_central_x_errs)
        run2_align_central_x_errs_med = np.median(run2_align_central_x_errs)
        
        run1_align_central_y_errs =  run1_align_y_errs[central_arcsec_filt_run1]
        run2_align_central_y_errs =  run2_align_y_errs[central_arcsec_filt_run2]
    
        run1_align_central_y_errs_med = np.median(run1_align_central_y_errs)
        run2_align_central_y_errs_med = np.median(run2_align_central_y_errs)
    
        run1_align_central_x_errs_mad = stats.median_abs_deviation(run1_align_central_x_errs)
        run2_align_central_x_errs_mad = stats.median_abs_deviation(run2_align_central_x_errs)
        
        run1_align_central_y_errs_mad = stats.median_abs_deviation(run1_align_central_y_errs)
        run2_align_central_y_errs_mad = stats.median_abs_deviation(run2_align_central_y_errs)
        
    
        print(run1_num_central_dets)
        print(run2_num_central_dets)
        
        # Create output table with overall stats
        stats_table = Table(
            [
                [self.run1_file_name_str, self.run2_file_name_str],
                [self.run1_name_str, self.run2_name_str],
                [run1_num_dets, run2_num_dets],
                [run1_num_central_dets, run2_num_central_dets],
                [run1_num_bright_dets, run2_num_bright_dets],
                [run1_stf_x_errs_med, run2_stf_x_errs_med],
                [run1_stf_x_errs_mad, run2_stf_x_errs_mad],
                [run1_stf_y_errs_med, run2_stf_y_errs_med],
                [run1_stf_y_errs_mad, run2_stf_y_errs_mad],
                [1000*run1_align_x_errs_med, 1000*run2_align_x_errs_med],
                [1000*run1_align_x_errs_mad, 1000*run2_align_x_errs_mad],
                [1000*run1_align_y_errs_med, 1000*run2_align_y_errs_med],
                [1000*run1_align_y_errs_mad, 1000*run2_align_y_errs_mad],
                [run1_stf_m_errs_med, run2_stf_m_errs_med],
                [run1_stf_m_errs_mad, run2_stf_m_errs_mad],
                [run1_stf_bright_x_errs_med, run2_stf_bright_x_errs_med],
                [run1_stf_bright_x_errs_mad, run2_stf_bright_x_errs_mad],
                [run1_stf_bright_y_errs_med, run2_stf_bright_y_errs_med],
                [run1_stf_bright_y_errs_mad, run2_stf_bright_y_errs_mad],
                [1000*run1_align_bright_x_errs_med, 1000*run2_align_bright_x_errs_med],
                [1000*run1_align_bright_x_errs_mad, 1000*run2_align_bright_x_errs_mad],
                [1000*run1_align_bright_y_errs_med, 1000*run2_align_bright_y_errs_med],
                [1000*run1_align_bright_y_errs_mad, 1000*run2_align_bright_y_errs_mad],
                [1000*run1_align_central_x_errs_med, 1000*run2_align_central_x_errs_med],
                [1000*run1_align_central_y_errs_med, 1000*run2_align_central_y_errs_med],
                [1000*run1_align_central_x_errs_mad, 1000*run2_align_central_x_errs_mad],
                [1000*run1_align_central_y_errs_mad, 1000*run2_align_central_y_errs_mad],
                [run1_stf_bright_m_errs_med, run2_stf_bright_m_errs_med],
                [run1_stf_bright_m_errs_mad, run2_stf_bright_m_errs_mad],
            ],
            names=(
                'run_file_name',
                'run_name',
                'num_detections',
                'num_central_detections',
                'num_bright_detections',
                'stf_x_err_pix_median',
                'stf_x_err_pix_median_abs_dev',
                'stf_y_err_pix_median',
                'stf_y_err_pix_median_abs_dev',
                'align_x_err_mas_median',
                'align_x_err_mas_median_abs_dev',
                'align_y_err_mas_median',
                'align_y_err_mas_median_abs_dev',
                'stf_mag_err_median',
                'stf_mag_err_median_abs_dev',
                'stf_bright_x_err_pix_median',
                'stf_bright_x_err_pix_median_abs_dev',
                'stf_bright_y_err_pix_median',
                'stf_bright_y_err_pix_median_abs_dev',
                'align_bright_x_err_mas_median',
                'align_bright_x_err_mas_median_abs_dev',
                'align_bright_y_err_mas_median',
                'align_bright_y_err_mas_median_abs_dev',
                'align_central_x_err_mas_median',
                'align_central_y_err_mas_median',
                'align_central_x_err_mas_median_abs_dev',
                'align_central_y_err_mas_median_abs_dev',
                'stf_bright_mag_err_median',
                'stf_bright_mag_err_median_abs_dev',
            ),
        )
        
        stats_table.write(
            out_dir + 'overall_stats.txt',
            format='ascii.fixed_width_two_line',
            overwrite=True,
            formats={
                'stf_x_err_pix_median': '%.3f',
                'stf_x_err_pix_median_abs_dev': '%.3f',
                'stf_y_err_pix_median': '%.3f',
                'stf_y_err_pix_median_abs_dev': '%.3f',
                'align_x_err_mas_median': '%.3f',
                'align_x_err_mas_median_abs_dev': '%.3f',
                'align_y_err_mas_median': '%.3f',
                'align_y_err_mas_median_abs_dev': '%.3f',
                'stf_mag_err_median': '%.3f',
                'stf_mag_err_median_abs_dev': '%.3f',
                'stf_bright_x_err_pix_median': '%.3f',
                'stf_bright_x_err_pix_median_abs_dev': '%.3f',
                'stf_bright_y_err_pix_median': '%.3f',
                'stf_bright_y_err_pix_median_abs_dev': '%.3f',
                'align_bright_x_err_mas_median': '%.3f',
                'align_bright_x_err_mas_median_abs_dev': '%.3f',
                'align_bright_y_err_mas_median': '%.3f',
                'align_bright_y_err_mas_median_abs_dev': '%.3f',
                'align_central_x_err_mas_median': '%.3f',
                'align_central_y_err_mas_median': '%.3f',
                'align_central_x_err_mas_median_abs_dev': '%.3f',
                'align_central_y_err_mas_median_abs_dev': '%.3f',
                'stf_bright_mag_err_median': '%.3f',
                'stf_bright_mag_err_median_abs_dev': '%.3f',
            },
        )
        stats_table.write(
            out_dir + 'overall_stats.h5',
            format='hdf5', path='data', serialize_meta=True,
            overwrite=True,
        )
        
        return stats_table
    
    
    def analyze_pos_comparison_rms(
        self,
        mag_bin_lo = -1, mag_bin_hi = -1,
        num_near_neighbors=10,
    ):
        cur_wd = os.getcwd()
        
        starlist_align_location = self.align_rms_dir + '/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'
        
        plot_out_dir = self.epoch_analysis_location + 'align_rms_pos_comparison_plots/'
        os.makedirs(plot_out_dir, exist_ok=True)
        
        out_mag_suffix = ''
        if mag_bin_lo != -1:
            out_mag_suffix = f'_mag_{mag_bin_lo}_{mag_bin_hi}'
        
        # Read in STF tables
        orig_run1_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt,
        )
        
        stf_run1_lis_table = stf_rms_lis_reader(orig_run1_stf_file)
        stf_run2_lis_table = stf_rms_lis_reader(orig_run2_stf_file)
        
        # Read in align tables
        align_mag_table = align_mag_reader(
            align_root + '.mag',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_pos_table = align_pos_reader(
            align_root + '.pos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_pos_err_table = align_pos_err_reader(
            align_root + '.err',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_orig_pos_table = align_orig_pos_reader(
            align_root + '.origpos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_param_table = align_param_reader(
            align_root + '.param',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        # Common detections in each starlist
        det_both = np.logical_and(
            align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
            align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
        )
        
        detection_filter = np.where(det_both)
        
        if mag_bin_lo != -1:
            det_mag = np.logical_and(
                align_mag_table[self.run2_file_name_str + '_mag'] >= mag_bin_lo,
                align_mag_table[self.run2_file_name_str + '_mag'] < mag_bin_hi,
            )
        
            detection_filter = np.where(np.logical_and(det_both, det_mag))
        
        common_detections_mag_table = align_mag_table[detection_filter]
        common_detections_pos_table = align_pos_table[detection_filter]
        common_detections_pos_err_table = align_pos_err_table[detection_filter]
        common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
        common_detections_param_table = align_param_table[detection_filter]
        
        ## Place to add like a combined starlist table /master table as well
        
        
        # Are align positions for run1 and run2 same? 
        common_detections_r2d_table = np.hypot(
            common_detections_pos_table[self.run1_file_name_str + '_x'],
            common_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        # One mode detections
        run1_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] == 0.0,
            ),
        )
        run2_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] == 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
            ),
        )
        
        run1_only_detections_mag_table = align_mag_table[run1_detection_filter]
        run1_only_detections_pos_table = align_pos_table[run1_detection_filter]
        run1_only_detections_pos_err_table = align_pos_err_table[run1_detection_filter]
        run1_only_detections_orig_pos_table = align_orig_pos_table[run1_detection_filter]
        run1_only_detections_param_table = align_param_table[run1_detection_filter]
        
        run1_only_detections_r2d_table = np.hypot(
            run1_only_detections_pos_table[self.run1_file_name_str + '_x'],
            run1_only_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        run2_only_detections_mag_table = align_mag_table[run2_detection_filter]
        run2_only_detections_pos_table = align_pos_table[run2_detection_filter]
        run2_only_detections_pos_err_table = align_pos_err_table[run2_detection_filter]
        run2_only_detections_orig_pos_table = align_orig_pos_table[run2_detection_filter]
        run2_only_detections_param_table = align_param_table[run2_detection_filter]
        
        run2_only_detections_r2d_table = np.hypot(
            run2_only_detections_pos_table[self.run2_file_name_str + '_x'],
            run2_only_detections_pos_table[self.run2_file_name_str + '_y'],
        )
        
        # Bright stars for reference
        if mag_bin_lo == -1:
            bright_cutoff = 12.
        
            bright_filter = np.where(common_detections_mag_table[self.run2_file_name_str + '_mag'] <= bright_cutoff)
        
            bright_pos_table = common_detections_pos_table[bright_filter]
            bright_orig_pos_table = common_detections_orig_pos_table[bright_filter]
        
        # Position differences
        diff_pos_x = (common_detections_orig_pos_table[self.run2_file_name_str + '_x'] -
                      common_detections_orig_pos_table[self.run1_file_name_str + '_x'])
        diff_pos_y = (common_detections_orig_pos_table[self.run2_file_name_str + '_y'] -
                      common_detections_orig_pos_table[self.run1_file_name_str + '_y'])
        
        common_x = common_detections_orig_pos_table[self.run1_file_name_str + '_x']
        common_y = common_detections_orig_pos_table[self.run1_file_name_str + '_y']
        
        if len(diff_pos_x) < 1:
            print('No stars in common between the starlists found')
            
            return
        
        # # Draw the comparison plots
        # # Position comparison
        # fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
        
        # ax1.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        # ax1.plot(common_detections_orig_pos_table[self.run1_file_name_str + '_x'],
        #          common_detections_orig_pos_table[self.run2_file_name_str + '_x'],
        #          '.', color='k', alpha=0.6)
        
        # ax2.plot(common_detections_orig_pos_table[self.run1_file_name_str + '_y'],
        #          common_detections_orig_pos_table[self.run2_file_name_str + '_y'],
        #          '.', color='k', alpha=0.6)
        
        # ax1.set_xlabel(self.run1_name_str + r": $x$")
        # ax1.set_ylabel(self.run2_name_str + r": $x$")
        
        # ax2.set_xlabel(self.run1_name_str + r": $y$")
        # ax2.set_ylabel(self.run2_name_str + r": $y$")
        
        # ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        
        # fig.tight_layout()
        
        # fig.savefig(
        #     f"{plot_out_dir}stf_pos_comparison{out_mag_suffix}.pdf",
        # )
        # fig.savefig(
        #     f"{plot_out_dir}stf_pos_comparison{out_mag_suffix}.png",
        #     dpi=200,
        # )
        
        # plt.close(fig)
        
        # # Delta position comparison
        # fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
        
        # ax1.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        # ax1.plot(common_detections_orig_pos_table[self.run1_file_name_str + '_x'],
        #          diff_pos_x,
        #          '.', color='k', alpha=0.6)
        
        # ax2.plot(common_detections_orig_pos_table[self.run1_file_name_str + '_y'],
        #          diff_pos_y,
        #          '.', color='k', alpha=0.6)
        
        # ax1.set_xlabel(self.run1_name_str + r": $x$")
        # ax1.set_ylabel(self.run1_name_str + r" $x$ $-$ " + self.run2_name_str + r" $x$")
        
        # ax2.set_xlabel(self.run1_name_str + r": $y$")
        # ax2.set_ylabel(self.run1_name_str + r" $y$ $-$ " + self.run2_name_str + r" $y$")
        
        # ax1.set_ylim([np.median(diff_pos_x) - 5.*stats.median_abs_deviation(diff_pos_x),
        #               np.median(diff_pos_x) + 5.*stats.median_abs_deviation(diff_pos_x)])
        
        # ax1.axhline(np.median(diff_pos_x), color='k', ls='-')
        # ax1.axhline(np.median(diff_pos_x) + stats.median_abs_deviation(diff_pos_x), color='k', ls='--')
        # ax1.axhline(np.median(diff_pos_x) - stats.median_abs_deviation(diff_pos_x), color='k', ls='--')
        
        # ax2.set_ylim([np.median(diff_pos_y) - 5.*stats.median_abs_deviation(diff_pos_y),
        #               np.median(diff_pos_y) + 5.*stats.median_abs_deviation(diff_pos_y)])
        
        # ax2.axhline(np.median(diff_pos_y), color='k', ls='-')
        # ax2.axhline(np.median(diff_pos_y) + stats.median_abs_deviation(diff_pos_y), color='k', ls='--')
        # ax2.axhline(np.median(diff_pos_y) - stats.median_abs_deviation(diff_pos_y), color='k', ls='--')
        
        
        # fig.tight_layout()
        
        # fig.savefig(f"{plot_out_dir}stf_pos_delta_comparison{out_mag_suffix}.pdf")
        # fig.savefig(f"{plot_out_dir}stf_pos_delta_comparison{out_mag_suffix}.png", dpi=200)
        
        # plt.close(fig)

        # Position error comparison
        fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
        
        ax1.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $x$')
        ax2.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $y$')
        
        ax1.plot(common_detections_pos_err_table[self.run1_file_name_str + '_x_err'],
                 (common_detections_pos_err_table[self.run2_file_name_str + '_x_err'] /
                  common_detections_pos_err_table[self.run1_file_name_str + '_x_err']),
                 '.', color='k', alpha=0.6)
        
        ax2.plot(common_detections_pos_err_table[self.run1_file_name_str + '_y_err'],
                 (common_detections_pos_err_table[self.run2_file_name_str + '_y_err'] /
                  common_detections_pos_err_table[self.run1_file_name_str + '_y_err']),
                 '.', color='k', alpha=0.6)
        
        # print(common_detections_pos_err_table[self.run1_file_name_str + '_y'])
        # print(common_detections_pos_err_table[self.run2_file_name_str + '_y'])
        
        ax1.set_xlabel(self.run1_name_str + r" STF $\sigma_x$ (arcsec)")
        ax1.set_ylabel(self.run2_name_str + r" STF $\sigma_x$ $/$ " + self.run1_name_str + r" STF $\sigma_x$")
        
        ax2.set_xlabel(self.run1_name_str + r" STF $\sigma_y$ (arcsec)")
        ax2.set_ylabel(self.run2_name_str + r" STF $\sigma_y$ $/$ " + self.run1_name_str + r" STF $\sigma_y$")
        
        # ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        
        plot_lims = [1e-5, 2e-2]
        y_plot_lims = [1e-2, 1e2]
        
        ax1.set_xlim(plot_lims)
        ax1.set_ylim(y_plot_lims)
        
        ax2.set_xlim(plot_lims)
        ax2.set_ylim(y_plot_lims)
        
        ax1.plot(plot_lims, [1, 1], 'k:')
        ax2.plot(plot_lims, [1, 1], 'k:')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        fig.tight_layout()
        
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison{out_mag_suffix}.pdf")
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison{out_mag_suffix}.png", dpi=200)
        
        plt.close(fig)
    
        if mag_bin_lo == -1:
            bright_cutoff = 12
            bright_filter = np.where(common_detections_mag_table[self.run2_file_name_str + '_mag'] <= bright_cutoff)
            bright_pos_err_table = common_detections_pos_err_table[bright_filter]
            
        ### fix this for other mag_bin_lo
        bright_cutoff = 12
        bright_filter = np.where(common_detections_mag_table[self.run2_file_name_str + '_mag'] <= bright_cutoff)
        bright_pos_err_table = common_detections_pos_err_table[bright_filter]
            
        # Position error comparison for only bright stars
        
        fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
        
        ax1.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $x$')
        ax2.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $y$')
        ax1.plot( bright_pos_err_table[self.run1_file_name_str + '_x_err'],
                 ( bright_pos_err_table[self.run2_file_name_str + '_x_err'] /
                   bright_pos_err_table[self.run1_file_name_str + '_x_err']),
                 '.', color='blue', alpha=0.6)
        
        ax2.plot( bright_pos_err_table[self.run1_file_name_str + '_y_err'],
                 ( bright_pos_err_table[self.run2_file_name_str + '_y_err'] /
                  bright_pos_err_table[self.run1_file_name_str + '_y_err']),
                 '.', color='blue', alpha=0.6)
        
        # print(common_detections_pos_err_table[self.run1_file_name_str + '_y'])
        # print(common_detections_pos_err_table[self.run2_file_name_str + '_y'])
        
        ax1.set_xlabel(self.run1_name_str + r" STF $\sigma_x$ (arcsec)")
        ax1.set_ylabel(self.run2_name_str + r" STF $\sigma_x$ $/$ " + self.run1_name_str + r" STF $\sigma_x$")
        
        ax2.set_xlabel(self.run1_name_str + r" STF $\sigma_y$ (arcsec)")
        ax2.set_ylabel(self.run2_name_str + r" STF $\sigma_y$ $/$ " + self.run1_name_str + r" STF $\sigma_y$")
        
        # ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        
        plot_lims = [1e-5, 2e-2]
        y_plot_lims = [1e-2, 1e2]
        
        ax1.set_xlim(plot_lims)
        ax1.set_ylim(y_plot_lims)
        
        ax2.set_xlim(plot_lims)
        ax2.set_ylim(y_plot_lims)
        
        ax1.plot(plot_lims, [1, 1], 'k:')
        ax2.plot(plot_lims, [1, 1], 'k:')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        fig.tight_layout()
        
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison_bright{out_mag_suffix}.pdf")
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison_bright{out_mag_suffix}.png", dpi=200)
        
        plt.close(fig)
        
        # Position error comparison for only central arcsecond stars
        
        central_arcsec_filt = np.where(common_detections_r2d_table < 1)
        common_detections_pos_err_central_arcsec_table = common_detections_pos_err_table[central_arcsec_filt]
        
        # Position error comparison
        fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
        
        ax1.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $x$')
        ax2.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt] + r' $y$')
        
        ax1.plot( common_detections_pos_err_central_arcsec_table[self.run1_file_name_str + '_x_err'],
                 ( common_detections_pos_err_central_arcsec_table[self.run2_file_name_str + '_x_err'] /
                   common_detections_pos_err_central_arcsec_table[self.run1_file_name_str + '_x_err']),
                 '.', color='k', alpha=0.6)
        
        ax2.plot( common_detections_pos_err_central_arcsec_table[self.run1_file_name_str + '_y_err'],
                 ( common_detections_pos_err_central_arcsec_table[self.run2_file_name_str + '_y_err'] /
                   common_detections_pos_err_central_arcsec_table[self.run1_file_name_str + '_y_err']),
                 '.', color='k', alpha=0.6)
        
        # print(common_detections_pos_err_table[self.run1_file_name_str + '_y'])
        # print(common_detections_pos_err_table[self.run2_file_name_str + '_y'])
        
        ax1.set_xlabel(self.run1_name_str + r" STF $\sigma_x$ (arcsec)")
        ax1.set_ylabel(self.run2_name_str + r" STF $\sigma_x$ $/$ " + self.run1_name_str + r" STF $\sigma_x$")
        
        ax2.set_xlabel(self.run1_name_str + r" STF $\sigma_y$ (arcsec)")
        ax2.set_ylabel(self.run2_name_str + r" STF $\sigma_y$ $/$ " + self.run1_name_str + r" STF $\sigma_y$")
        
        # ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        
        plot_lims = [1e-5, 2e-2]
        y_plot_lims = [1e-2, 1e2]
        
        ax1.set_xlim(plot_lims)
        ax1.set_ylim(y_plot_lims)
        
        ax2.set_xlim(plot_lims)
        ax2.set_ylim(y_plot_lims)
        
        ax1.plot(plot_lims, [1, 1], 'k:')
        ax2.plot(plot_lims, [1, 1], 'k:')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        fig.tight_layout()
        
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison_central{out_mag_suffix}.pdf")
        fig.savefig(f"{plot_out_dir}stf_pos_err_comparison_central{out_mag_suffix}.png", dpi=200)
        
        plt.close(fig)
        
        # Quiver plot
        fig, ax = plt.subplots(figsize=(4.8,5.2))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        q = ax.quiver(common_detections_pos_table[self.run1_file_name_str + '_x'] * -1,
                      common_detections_pos_table[self.run1_file_name_str + '_y'],
                      diff_pos_x, diff_pos_y,
                      scale=0.25, scale_units='xy')
        
        ax.quiverkey(q, X=0.05, Y=-0.15, U=0.1, labelpos='E',
                     label=r'0.1 pixel, ' + self.run2_name_str + r' $-$ ' + self.run1_name_str + r' Starfinder position',
                     fontproperties={'size':'x-small'})
        
        if mag_bin_lo == -1:
            ax.plot(bright_pos_table[self.run1_file_name_str + '_x'] * -1,
                    bright_pos_table[self.run1_file_name_str + '_y'],
                    'o', color='royalblue', ms=2.0)
                
        ax.set_aspect('equal', 'box')
        
        ax.set_xlabel(r"arcsec E of Sgr A*")
        ax.set_ylabel(r"arcsec N of Sgr A*")
        
        ax.set_xlim([6, -6])
        ax.set_ylim([-7, 5])
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(2)
        y_minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison{out_mag_suffix}.pdf')
        fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison{out_mag_suffix}.png', dpi=200)
        
        plt.close(fig)
        
        # Nearest neighbors plot
        ## Construct kd-tree for quick lookup for closest neighbors
        neighbor_kdtree = KDTree(list(zip(common_x, common_y)))
        
        ## Construct grid points for nearest neighbors map
        near_neighbors_map_x_bounds = [0, 1200]
        near_neighbors_map_y_bounds = [0, 1200]
        
        near_neighbors_map_grid_spacing = 20
        
        near_neighbors_map_x_coords = np.arange(near_neighbors_map_x_bounds[0], near_neighbors_map_x_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)
        near_neighbors_map_y_coords = np.arange(near_neighbors_map_y_bounds[0], near_neighbors_map_y_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)
        
        near_neighbors_map_plot_x, near_neighbors_map_plot_y = np.meshgrid(near_neighbors_map_x_coords, near_neighbors_map_y_coords)
        
        ## Go through each grid point
        ### Function to evaluate mean and median value of nearest neighbors
        def near_neighbors_mean_median_val(x_coord, y_coord):
            #### Find nearest neighbor stars
            near_neighbors = neighbor_kdtree.query([x_coord, y_coord], k=num_near_neighbors)
            near_neighbors_coords, near_neighbors_indices = near_neighbors
        
            #### Compute mean reduced chi squared of the nearest neighbors
            val_x_array = np.empty(len(near_neighbors_indices))
            val_y_array = np.empty(len(near_neighbors_indices))
        
            for index in range(len(near_neighbors_indices)):
                cur_neighbor_index = near_neighbors_indices[index]
            
                val_x_array[index] = diff_pos_x[cur_neighbor_index]
                val_y_array[index] = diff_pos_y[cur_neighbor_index]
        
            mean_val_x = np.mean(val_x_array)
            median_val_x = np.median(val_x_array)
        
            mean_val_y = np.mean(val_y_array)
            median_val_y = np.median(val_y_array)
        
            return (mean_val_x, median_val_x,
                    mean_val_y, median_val_y)
        
        ### Vectorized version of near_neighbors_mean_median_rcs function
        vector_near_neighbors_mean_median_val = np.vectorize(near_neighbors_mean_median_val)
        
        ### Evaluate over all grid points
        (near_neighbors_map_plot_mean_val_x,
         near_neighbors_map_plot_median_val_x,
         near_neighbors_map_plot_mean_val_y,
         near_neighbors_map_plot_median_val_y) = vector_near_neighbors_mean_median_val(
                                                   near_neighbors_map_plot_x,
                                                   near_neighbors_map_plot_y)
        
        # Compute a transformation from pixels to abs coordinates
        # using nearest star to Sgr A*
        rad_distance = np.hypot(
            common_detections_pos_table[self.run1_file_name_str + '_x'],
            common_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        close_star_index = np.argmin(rad_distance)
        close_star_op_x = (common_detections_orig_pos_table[self.run1_file_name_str + '_x'])[close_star_index]
        close_star_op_y = (common_detections_orig_pos_table[self.run1_file_name_str + '_y'])[close_star_index]
        close_star_pos_x = (common_detections_pos_table[self.run1_file_name_str + '_x'])[close_star_index]
        close_star_pos_y = (common_detections_pos_table[self.run1_file_name_str + '_y'])[close_star_index]
        
        center_op_x = close_star_op_x - (close_star_pos_x/self.plate_scale)
        center_op_y = close_star_op_y - (close_star_pos_y/self.plate_scale)
        
        
        # Quiver plot
        fig, ax = plt.subplots(figsize=(4.8,5.2))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        q = ax.quiver((near_neighbors_map_plot_x - center_op_x) * self.plate_scale * -1,
                      (near_neighbors_map_plot_y - center_op_y) * self.plate_scale,
                      near_neighbors_map_plot_median_val_x,
                      near_neighbors_map_plot_median_val_y,
                      scale=0.25, scale_units='xy')
        
        quiver_label = r'0.1 pixel, ' + self.run2_name_str + r' $-$ ' + self.run1_name_str + r' Starfinder position'
        quiver_label += ' (median of 20 nearest neighbors)'
        
        ax.quiverkey(q, X=0.05, Y=-0.15, U=0.1, labelpos='E',
                     label=quiver_label,
                     fontproperties={'size':'x-small'})
        
        if mag_bin_lo == -1:
            ax.plot(bright_pos_table[self.run1_file_name_str + '_x'] * -1,
                    bright_pos_table[self.run1_file_name_str + '_y'],
                    'o', color='royalblue', ms=2.5)
        
        ax.set_aspect('equal', 'box')
        
        ax.set_xlabel(r"arcsec E of Sgr A*")
        ax.set_ylabel(r"arcsec N of Sgr A*")
        
        ax.set_xlim([6, -6])
        ax.set_ylim([-7, 5])
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(2)
        y_minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        
        # Save out and close figure
        fig.tight_layout()
        
        out_mag_suffix = ''
        if mag_bin_lo != -1:
            out_mag_suffix = f'_mag_{mag_bin_lo}_{mag_bin_hi}'
        
        fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison_nearneigh{out_mag_suffix}.pdf')
        fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison_nearneigh{out_mag_suffix}.png', dpi=200)
        
        plt.close(fig)
        
        # Draw r2d histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.arange(0, 5, 0.75)
        
        (common_hist_r2d, common_bins) = np.histogram(
            common_detections_r2d_table,
            bins=hist_bins,
        )
        (run1_only_hist_r2d, run1_only_bins) = np.histogram(
            run1_only_detections_r2d_table,
            bins=hist_bins,
        )
        (run2_only_hist_r2d, run2_only_bins) = np.histogram(
            run2_only_detections_r2d_table,
            bins=hist_bins,
        )
        
        # ax.hist(common_detections_r2d_table,
        #         bins=hist_bins,
        #         histtype='step',
        #         color='k', label='Detections in both modes')
        
        ax.hist(
            hist_bins[:-1],
            bins=hist_bins,
            weights=run1_only_hist_r2d/common_hist_r2d,
            histtype='step',
            color='C0',
            label=self.run1_name_str + ' only detections',
        )
        
        ax.hist(
            hist_bins[:-1],
            bins=hist_bins,
            weights=run2_only_hist_r2d/common_hist_r2d,
            histtype='step',
            color='C1',
            label=self.run2_name_str + ' only detections',
        )
        
        ax.set_xlabel(r'Distance from Sgr A* (arcsec)')
        ax.set_ylabel('Unique Dets. / Common Dets.')
        
        ax.legend(loc='upper right')
        
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(1)
        x_minorLocator = MultipleLocator(0.25)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(.1)
        y_minorLocator = MultipleLocator(.02)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'pos_num_hist.pdf')
        fig.savefig(plot_out_dir + 'pos_num_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw corr comparison
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.plot(
            common_detections_param_table[self.run1_file_name_str + '_corr'],
            common_detections_param_table[self.run2_file_name_str + '_corr'],
            'ko', alpha=0.2,
        )
        
        ax.set_xlabel(f'{self.run1_name_str} Correlation')
        ax.set_ylabel(f'{self.run2_name_str} Correlation')
        
        # ax.legend(loc='upper left')
        
        ax.set_xlim([0.79, 1.01])
        ax.set_ylim([0.79, 1.01])
        
        x_majorLocator = MultipleLocator(0.05)
        x_minorLocator = MultipleLocator(0.01)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.05)
        y_minorLocator = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'corr_comparison.pdf')
        fig.savefig(plot_out_dir + 'corr_comparison.png', dpi=200)
        
        plt.close(fig)
        
        # Draw common stars' corr histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.linspace(0.8, 1.0, num=11)
        
        # (common_hist_mag, common_bins) = np.histogram(
        #     common_detections_mag_table[single_version_str + '_mag'],
        #     bins=hist_bins)
        # (legonly_hist_mag, legonly_bins) = np.histogram(
        #     legonly_detections_mag_table[legacy_version_str + '_mag'],
        #     bins=hist_bins
        # )
        # (sinonly_hist_mag, sinonly_bins) = np.histogram(
        #     sinonly_detections_mag_table[single_version_str + '_mag'],
        #     bins=hist_bins
        # )
        #
        # print(common_hist_mag)
        # print(legonly_hist_mag)
        
        # ax.hist(hist_bins[:-1],
        #         bins=hist_bins,
        #         weights=common_hist_mag,
        #         histtype='step',
        #         color='k', label='Detections in both modes')
        
        ax.hist(
            common_detections_param_table[self.run1_file_name_str + '_corr'],
            bins=hist_bins,
            histtype='step',
            color='C0',
            label=f'Common detections, {self.run1_name_str} correlation',
        )
        
        ax.hist(
            common_detections_param_table[self.run2_file_name_str + '_corr'],
            bins=hist_bins,
            histtype='step',
            color='C1',
            label=f'Common detections, {self.run2_name_str} correlation',
        )
        
        ax.set_xlabel(r'Correlation')
        ax.set_ylabel('Detections')
        
        ax.legend(loc='upper left')
        
        # ax.set_xlim([9.5, 20.5])
        # ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(0.05)
        x_minorLocator = MultipleLocator(0.01)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        # y_majorLocator = MultipleLocator(50)
        # y_minorLocator = MultipleLocator(10)
        # ax.yaxis.set_major_locator(y_majorLocator)
        # ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'common_corr_hist.pdf')
        fig.savefig(plot_out_dir + 'common_corr_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw unique detections' corr histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.linspace(0.8, 1.0, num=11)
        
        # (common_hist_mag, common_bins) = np.histogram(
        #     common_detections_mag_table[single_version_str + '_mag'],
        #     bins=hist_bins)
        # (legonly_hist_mag, legonly_bins) = np.histogram(
        #     legonly_detections_mag_table[legacy_version_str + '_mag'],
        #     bins=hist_bins
        # )
        # (sinonly_hist_mag, sinonly_bins) = np.histogram(
        #     sinonly_detections_mag_table[single_version_str + '_mag'],
        #     bins=hist_bins
        # )
        #
        # print(common_hist_mag)
        # print(legonly_hist_mag)
        
        # ax.hist(hist_bins[:-1],
        #         bins=hist_bins,
        #         weights=common_hist_mag,
        #         histtype='step',
        #         color='k', label='Detections in both modes')
        
        ax.hist(
            run1_only_detections_param_table[self.run1_file_name_str + '_corr'],
            bins=hist_bins,
            histtype='step',
            color='C0',
            label=f'Unique detections, {self.run1_name_str} correlation',
        )
        
        ax.hist(
            run2_only_detections_param_table[self.run2_file_name_str + '_corr'],
            bins=hist_bins,
            histtype='step',
            color='C1',
            label=f'Unique detections, {self.run2_name_str} correlation',
        )
        
        ax.set_xlabel(r'Correlation')
        ax.set_ylabel('Detections')
        
        ax.legend(loc='upper left')
        
        # ax.set_xlim([9.5, 20.5])
        # ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(0.05)
        x_minorLocator = MultipleLocator(0.01)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        # y_majorLocator = MultipleLocator(50)
        # y_minorLocator = MultipleLocator(10)
        # ax.yaxis.set_major_locator(y_majorLocator)
        # ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'unique_corr_hist.pdf')
        fig.savefig(plot_out_dir + 'unique_corr_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw all corr histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.arange(0.8, 1.2, 0.02)
        
        (common_hist_corr, common_bins) = np.histogram(
            (common_detections_param_table[self.run1_file_name_str + '_corr'] +\
            common_detections_param_table[self.run2_file_name_str + '_corr'])/2.,
            bins=hist_bins,
        )
        (run1_only_hist_corr, run1_only_bins) = np.histogram(
            run1_only_detections_param_table[self.run1_file_name_str + '_corr'],
            bins=hist_bins
        )
        (run2_only_hist_corr, run2_only_bins) = np.histogram(
            run2_only_detections_param_table[self.run2_file_name_str + '_corr'],
            bins=hist_bins
        )
        #
        # print(common_hist_mag)
        # print(legonly_hist_mag)
        
        ax.hist(
            hist_bins[:-1],
            bins=hist_bins,
            weights=common_hist_corr,
            histtype='step',
            color='k',
            label='Detections in both modes',
        )
        
        ax.hist(
            hist_bins[:-1],
            bins=hist_bins,
            weights=run1_only_hist_corr,
            histtype='step',
            color='C0',
            label=self.run1_name_str + ' only detections',
        )
        
        ax.hist(
            hist_bins[:-1],
            bins=hist_bins,
            weights=run2_only_hist_corr,
            histtype='step',
            color='C1',
            label=self.run2_name_str + ' only detections',
        )
        
        ax.set_xlabel(r'Correlation')
        ax.set_ylabel(r'Detections')
        
        ax.legend(loc='upper left')
        
        ax.set_xlim([0.8, 1.0])
        # ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(0.05)
        x_minorLocator = MultipleLocator(0.01)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        # y_majorLocator = MultipleLocator(50)
        # y_minorLocator = MultipleLocator(10)
        # ax.yaxis.set_major_locator(y_majorLocator)
        # ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'corr_hist.pdf')
        fig.savefig(plot_out_dir + 'corr_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw common correlation ratio vs. avg plot
        common_corr_ratio =\
            common_detections_param_table[self.run2_file_name_str + '_corr'] /\
            common_detections_param_table[self.run1_file_name_str + '_corr']
        common_corr_avg =\
            (common_detections_param_table[self.run2_file_name_str + '_corr'] +
             common_detections_param_table[self.run1_file_name_str + '_corr'])/2.
        
        median_ratio = np.median(common_corr_ratio)
        std_sqrt_ratio = np.std(common_corr_ratio) / np.sqrt(len(common_corr_ratio))
        
        print(median_ratio)
        print(std_sqrt_ratio)
        
        fig, ax = plt.subplots(figsize=(6,3))
        
        ax.plot(common_corr_avg,
                common_corr_ratio,
                'k.', alpha=0.2)
        
        # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
        
        ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
                   label='Median Ratio')
        
        ax.fill_between(
            [0.75, 1.05],
            y1=[median_ratio + std_sqrt_ratio],
            y2=[median_ratio - std_sqrt_ratio],
            color='C4')
        
        # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
        #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
        
        ax.set_xlabel('Avg. Detection Correlation')
        ax.set_ylabel(f'{self.run2_name_str} Corr. / {self.run1_name_str} Corr.')
        
        ax.legend(loc='lower left')
        
        ax.set_xlim([0.78, 1.02])
        ax.set_ylim([0.75, 1.25])
        
        x_majorLocator = MultipleLocator(0.025)
        x_minorLocator = MultipleLocator(0.005)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.1)
        y_minorLocator = MultipleLocator(0.02)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'common_corr_ratio_avg.pdf')
        fig.savefig(plot_out_dir + 'common_corr_ratio_avg.png', dpi=200)
        
        plt.close(fig)
        
        # Draw common correlation ratio vs. r2d    
        fig, ax = plt.subplots(figsize=(6,3))
        
        ax.plot(common_detections_r2d_table,
                common_corr_ratio,
                'k.', alpha=0.2)
        
        # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
        
        # ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
        #            label='Median Ratio')
        #
        # ax.fill_between(
        #     [0.75, 1.05],
        #     y1=[median_ratio + std_sqrt_ratio],
        #     y2=[median_ratio - std_sqrt_ratio],
        #     color='C4')
        
        # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
        #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
        
        ax.set_xlabel('Distance from Sgr A* (arcsec)')
        ax.set_ylabel(f'{self.run2_name_str} Corr. / {self.run1_name_str} Corr.')
        
        # ax.legend(loc='lower left')
        
        ax.set_xlim([0, 7.5])
        ax.set_ylim([0.75, 1.25])
        
        x_majorLocator = MultipleLocator(2.0)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.1)
        y_minorLocator = MultipleLocator(0.02)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'common_corr_ratio_r2d.pdf')
        fig.savefig(plot_out_dir + 'common_corr_ratio_r2d.png', dpi=200)
        
        plt.close(fig)
        
        # Draw common correlation ratio vs. r2d    
        fig, ax = plt.subplots(figsize=(6,3), frameon=False)
        
        ax.plot(common_detections_r2d_table,
                common_corr_ratio,
                'k.', alpha=0.4)
        
        central_half_arcsec_filt = np.where(common_detections_r2d_table < 0.5)
        central_half_arcsec_cut = common_corr_ratio[central_half_arcsec_filt]
        
        central_half_arcsec_median_ratio = np.median(central_half_arcsec_cut)
        central_half_arcsec_std_sqrt_ratio = np.std(central_half_arcsec_cut) / \
            np.sqrt(len(central_half_arcsec_cut))
        
        # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
        
        ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
                   label='Median ratio, all stars')
        ax.fill_between(
            [0, 7.5],
            y1=[median_ratio + std_sqrt_ratio],
            y2=[median_ratio - std_sqrt_ratio],
            color='C4')
        
        ax.plot([0, 0.5],
                [central_half_arcsec_median_ratio, central_half_arcsec_median_ratio],
                color='C5', ls='-', lw=1.0, alpha=0.5,
                label='Median ratio, central half arcsecond stars')
        ax.fill_between(
            [0, 0.5],
            y1=[central_half_arcsec_median_ratio + central_half_arcsec_std_sqrt_ratio],
            y2=[central_half_arcsec_median_ratio - central_half_arcsec_std_sqrt_ratio],
            color='C5', alpha=0.5)
        
        # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
        #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
        
        ax.set_xlabel('Distance from Sgr A* (arcsec)')
        ax.set_ylabel(f'{self.run2_name_str} Corr. / {self.run1_name_str} Corr.')
        
        ax.legend(loc='lower right')
        
        ax.set_xlim([0, 2.0])
        ax.set_ylim([0.75, 1.25])
        
        x_majorLocator = MultipleLocator(0.5)
        x_minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.1)
        y_minorLocator = MultipleLocator(0.02)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'common_corr_ratio_r2d_central_half_arcsec.pdf')
        fig.savefig(plot_out_dir + 'common_corr_ratio_r2d_central_half_arcsec.png', dpi=200)
        
        plt.close(fig)
    
    
    def analyze_mag_comparison_rms(
        self,
    ):
        cur_wd = os.getcwd()
        
        starlist_align_location = self.align_rms_dir + '/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'
        
        plot_out_dir = self.epoch_analysis_location + 'align_rms_mag_comparison_plots/'
        os.makedirs(plot_out_dir, exist_ok=True)
        
        out_mag_suffix = ''
        
        # Read in STF tables
        orig_run1_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt,
        )
        
        stf_run1_lis_table = stf_rms_lis_reader(orig_run1_stf_file)
        stf_run2_lis_table = stf_rms_lis_reader(orig_run2_stf_file)
        
        # Read in align tables
        align_mag_table = align_mag_reader(
            align_root + '.mag',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_pos_table = align_pos_reader(
            align_root + '.pos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_orig_pos_table = align_orig_pos_reader(
            align_root + '.origpos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        # Common detections in each starlist
        det_both = np.logical_and(
            align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
            align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
        )
        
        detection_filter = np.where(det_both)
        
        common_detections_mag_table = align_mag_table[detection_filter]
        common_detections_pos_table = align_pos_table[detection_filter]
        common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
        
        common_detections_r2d_table = np.hypot(
            common_detections_pos_table[self.run1_file_name_str + '_x'],
            common_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        # One mode detections
        run1_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] != 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] == 0.0,
            ),
        )
        run2_detection_filter = np.where(
            np.logical_and(
                align_mag_table[self.run1_file_name_str + '_mag'] == 0.0,
                align_mag_table[self.run2_file_name_str + '_mag'] != 0.0,
            ),
        )
        
        run1_only_detections_mag_table = align_mag_table[run1_detection_filter]
        run1_only_detections_pos_table = align_pos_table[run1_detection_filter]
        run1_only_detections_orig_pos_table = align_orig_pos_table[run1_detection_filter]
        
        run1_only_detections_r2d_table = np.hypot(
            run1_only_detections_pos_table[self.run1_file_name_str + '_x'],
            run1_only_detections_pos_table[self.run1_file_name_str + '_y'],
        )
        
        run2_only_detections_mag_table = align_mag_table[run2_detection_filter]
        run2_only_detections_pos_table = align_pos_table[run2_detection_filter]
        run2_only_detections_orig_pos_table = align_orig_pos_table[run2_detection_filter]
        
        run2_only_detections_r2d_table = np.hypot(
            run2_only_detections_pos_table[self.run2_file_name_str + '_x'],
            run2_only_detections_pos_table[self.run2_file_name_str + '_y'],
        )
        
        # Mag differences and binned mag differences
        diff_mag = (
            common_detections_mag_table[self.run2_file_name_str + '_mag'] -
            common_detections_mag_table[self.run1_file_name_str + '_mag']
        )
        
        diff_mag_median_bin_cents = np.arange(9.5, 21., 0.5)
        diff_mag_median_bin_size = 0.5
        
        if self.epoch_filt == 'kp':
            diff_mag_median_bin_cents = diff_mag_median_bin_cents + 0.0
        elif self.epoch_filt == 'h':
            diff_mag_median_bin_cents = diff_mag_median_bin_cents + 2.0
        
        diff_mag_medians = 0. * diff_mag_median_bin_cents
        diff_mag_MADs = 0. * diff_mag_median_bin_cents
        
        diff_mag_MAD_hi = 0. * diff_mag_median_bin_cents
        diff_mag_MAD_lo = 0. * diff_mag_median_bin_cents
        
        for (cur_bin_cent,
             cur_bin_index) in zip(diff_mag_median_bin_cents,
                                   range(len(diff_mag_medians))):
            cur_bin_hi = cur_bin_cent + (diff_mag_median_bin_size/2.)
            cur_bin_lo = cur_bin_cent - (diff_mag_median_bin_size/2.)
            
            mag_bin_filter = np.where(
                np.logical_and(
                    common_detections_mag_table[self.run1_file_name_str + '_mag'] >= cur_bin_lo,
                    common_detections_mag_table[self.run1_file_name_str + '_mag'] < cur_bin_hi,
                ),
            )
            
            filtered_diff_mags = diff_mag[mag_bin_filter]
            
            diff_mag_medians[cur_bin_index] = np.median(filtered_diff_mags)
            diff_mag_MADs[cur_bin_index] = stats.median_abs_deviation(filtered_diff_mags)
            
            diff_mag_MAD_hi[cur_bin_index] = (diff_mag_medians[cur_bin_index] +
                                              diff_mag_MADs[cur_bin_index])
            diff_mag_MAD_lo[cur_bin_index] = (diff_mag_medians[cur_bin_index] -
                                              diff_mag_MADs[cur_bin_index])

        # Plot luminosity functions for each mode
        mag_hist_bins = np.arange(9.0, 22.5, 0.5)
        if self.epoch_filt == 'h':
            mag_hist_bins = mag_hist_bins + 2.0
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        ax.hist(
            stf_run1_lis_table['m'], bins=mag_hist_bins,
            histtype='step', color='C0', lw=1.5,
            label=self.run1_name_str,
        )
        
        ax.hist(
            stf_run2_lis_table['m'], bins=mag_hist_bins,
            histtype='step', color='C1',
            label=self.run2_name_str,
        )
        
        ax.legend(loc='upper left')
        
        ax.set_xlabel(self.filt_mag_label_strs[self.epoch_filt])
        ax.set_ylabel(r"Stars Detected")
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(50)
        y_minorLocator = MultipleLocator(10)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        ax.set_ylim([0, 300])
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'stf_lum_funcs.pdf')
        fig.savefig(plot_out_dir + 'stf_lum_funcs.png', dpi=200)
        
        plt.close(fig)
        
        # Plot luminosity function with one mode detections    
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        ax.hist(
            run1_only_detections_mag_table[self.run1_file_name_str + '_mag'],
            bins=mag_hist_bins,
            histtype='step', color='C0', lw=1.5,
            label=self.run1_name_str + " Only",
        )
        
        ax.hist(
            run2_only_detections_mag_table[self.run2_file_name_str + '_mag'],
            bins=mag_hist_bins,
            histtype='step', color='C1',
            label=self.run2_name_str + " Only",
        )
        
        ax.legend(loc='upper left')
        
        ax.set_xlabel(self.filt_mag_label_strs[self.epoch_filt])
        ax.set_ylabel(r"Stars Detected")
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(20)
        y_minorLocator = MultipleLocator(5)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        ax.set_ylim([0, 150])
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'stf_lum_funcs_one_mode.pdf')
        fig.savefig(plot_out_dir + 'stf_lum_funcs_one_mode.png', dpi=200)
        
        plt.close(fig)
        
        # Plot magnitude comparison
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        ax.plot(
            common_detections_mag_table[self.run1_file_name_str + '_mag'],
            common_detections_mag_table[self.run2_file_name_str + '_mag'],
            '.', color='royalblue', alpha=0.6,
        )
        
        # Diagonal line for comparison
        ax.plot(
            [np.min(mag_hist_bins), np.max(mag_hist_bins)],
            [np.min(mag_hist_bins), np.max(mag_hist_bins)],
            'k--', lw=0.5,
        )
        
        ax.set_xlabel(
            self.run1_name_str + ' StarFinder: ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        ax.set_ylabel(
            self.run2_name_str + ' StarFinder: ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(2)
        y_minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        ax.set_ylim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        
        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'stf_mag_comparison.pdf')
        fig.savefig(plot_out_dir + 'stf_mag_comparison.png', dpi=200)
        
        plt.close(fig)
        
        # Plot delta magnitude comparison
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        ax.plot(
            common_detections_mag_table[self.run1_file_name_str + '_mag'],
            diff_mag,
            '.', color='royalblue', alpha=0.6,
        )
        
        # Flat and binned median lines for comparison
        ax.fill_between(
            diff_mag_median_bin_cents,
            y1=diff_mag_MAD_lo, y2=diff_mag_MAD_hi,
            facecolor='r', edgecolor='none', alpha=0.5,
        )
        ax.plot(
            diff_mag_median_bin_cents,
            diff_mag_medians, 'r--', lw=0.5,
        )
        
        
        ax.plot(
            [np.min(mag_hist_bins), np.max(mag_hist_bins)],
            [0, 0], 'k--', lw=0.5,
        )
        
        
        ax.set_xlabel(
            self.run1_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        ax.set_ylabel(
            self.run2_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt] +\
            r" $-$ " +\
            self.run1_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        
        ax.text(10, -1.0, f'Brighter in {self.run2_name_str} Starfinder',
                fontsize='x-small', va='center')
        ax.text(10, 1.0, f'Fainter in {self.run2_name_str} Starfinder',
                fontsize='x-small', va='center')
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.5)
        y_minorLocator = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        ax.set_ylim([-1.5, 1.5])
        
        # ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'stf_mag_delta_comparison.pdf')
        fig.savefig(plot_out_dir + 'stf_mag_delta_comparison.png', dpi=200)
        
        plt.close(fig)
        
        
        # Plot delta magnitude comparison
        fig, ax = plt.subplots(figsize=(6,6))
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        # Flat and binned median lines for comparison
        ax.fill_between(
            diff_mag_median_bin_cents,
            y1=diff_mag_MAD_lo, y2=diff_mag_MAD_hi,
            facecolor='r', edgecolor='none', alpha=0.5,
        )
        ax.plot(
            diff_mag_median_bin_cents, diff_mag_medians, 'r--', lw=0.5,
        )
        
        ax.plot(
            [np.min(mag_hist_bins), np.max(mag_hist_bins)],
            [0, 0], 'k--', lw=0.5,
        )
        
        ax.set_xlabel(
            self.run1_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        ax.set_ylabel(
            self.run2_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt] +\
            r" $-$ " +\
            self.run1_name_str + ' ' + self.filt_mag_label_strs[self.epoch_filt]
        )
        
        ax.text(10, -1.0, f'Brighter in {self.run2_name_str} Starfinder',
                fontsize='x-small', va='center')
        ax.text(10, 1.0, f'Fainter in {self.run2_name_str} Starfinder',
                fontsize='x-small', va='center')
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(0.5)
        y_minorLocator = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        ax.set_ylim([-1.5, 1.5])
        
        # ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        
        # Save out and close figure
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'stf_mag_delta_comparison_nostars.pdf')
        fig.savefig(plot_out_dir + 'stf_mag_delta_comparison_nostars.png', dpi=200)
        
        plt.close(fig)
        
        
        # Draw mag histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        (common_hist_mag, common_bins) = np.histogram(
            common_detections_mag_table[self.run2_file_name_str + '_mag'],
            bins=mag_hist_bins,
        )
        (run1_only_hist_mag, run1_only_bins) = np.histogram(
            run1_only_detections_mag_table[self.run1_file_name_str + '_mag'],
            bins=mag_hist_bins,
        )
        (run2_only_hist_mag, run2_only_bins) = np.histogram(
            run2_only_detections_mag_table[self.run2_file_name_str + '_mag'],
            bins=mag_hist_bins,
        )
        
        # Write out table for the mag histogram
        mag_hist_table = Table(
            [
                mag_hist_bins[:-1],
                mag_hist_bins[1:],
                common_hist_mag,
                run1_only_hist_mag,
                run2_only_hist_mag,
            ],
            names=(
                'mag_bin_lo',
                'mag_bin_hi',
                'num_dets_common',
                f'num_dets_{self.run1_file_name_str}_only',
                f'num_dets_{self.run2_file_name_str}_only',
            )
        )
        
        mag_hist_table.write(
            plot_out_dir + 'mag_num_hist.txt',
            format='ascii.fixed_width_two_line',
            overwrite=True,
        )
        mag_hist_table.write(
            plot_out_dir + 'mag_num_hist.h5',
            format='hdf5', path='data', serialize_meta=True,
            overwrite=True,
        )
        
        # Draw plot
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=common_hist_mag,
            histtype='step',
            color='k', label='Detections in both modes',
        )
        
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=run1_only_hist_mag,
            histtype='step',
            hatch='///',
            color='C0',
            label=f'{self.run1_name_str} only detections',
        )
        
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=run2_only_hist_mag,
            histtype='step',
            hatch='\\\\\\',
            color='C1',
            label=f'{self.run2_name_str} only detections',
        )
        
        ax.set_xlabel(r'Mag')
        ax.set_ylabel('Detections')
        
        ax.legend(loc='upper left')
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        # ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(50)
        y_minorLocator = MultipleLocator(10)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'mag_num_hist.pdf')
        fig.savefig(plot_out_dir + 'mag_num_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw mag histogram for just the central arcsecond
        fig, ax = plt.subplots(figsize=(6,3))
        
        (common_hist_mag, common_bins) = np.histogram(
            (common_detections_mag_table[self.run2_file_name_str + '_mag'])[
                np.where(common_detections_r2d_table <= 1.0)
            ],
            bins=mag_hist_bins,
        )
        (run1_only_hist_mag, run1_only_bins) = np.histogram(
            (run1_only_detections_mag_table[self.run1_file_name_str + '_mag'])[
                np.where(run1_only_detections_r2d_table <= 1.0)
            ],
            bins=mag_hist_bins,
        )
        (run2_only_hist_mag, run2_only_bins) = np.histogram(
            (run2_only_detections_mag_table[self.run2_file_name_str + '_mag'])[
                np.where(run2_only_detections_r2d_table <= 1.0)
            ],
            bins=mag_hist_bins,
        )
        
        # Write out table for the central arcsecond histogram
        mag_hist_table = Table(
            [
                mag_hist_bins[:-1],
                mag_hist_bins[1:],
                common_hist_mag,
                run1_only_hist_mag,
                run2_only_hist_mag,
            ],
            names=(
                'mag_bin_lo',
                'mag_bin_hi',
                'num_dets_common',
                f'num_dets_{self.run1_file_name_str}_only',
                f'num_dets_{self.run2_file_name_str}_only',
            )
        )
        
        mag_hist_table.write(
            plot_out_dir + 'mag_num_hist_centarcsec.txt',
            format='ascii.fixed_width_two_line',
            overwrite=True,
        )
        mag_hist_table.write(
            plot_out_dir + 'mag_num_hist_centarcsec.h5',
            format='hdf5', path='data', serialize_meta=True,
            overwrite=True,
        )
        
        # Draw plot
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=common_hist_mag,
            histtype='step',
            color='k', label='Detections in both modes',
        )
        
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=run1_only_hist_mag,
            histtype='step',
            hatch='///',
            color='C0',
            label=f'{self.run1_name_str} only detections',
        )
        
        ax.hist(
            mag_hist_bins[:-1],
            bins=mag_hist_bins,
            weights=run2_only_hist_mag,
            histtype='step',
            hatch='\\\\\\',
            color='C1',
            label=f'{self.run2_name_str} only detections',
        )
        
        ax.set_xlabel(r'Mag')
        ax.set_ylabel('Central Arcsecond Detections')
        
        ax.legend(loc='upper left')
        
        ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
        # ax.set_ylim([0, 0.5])
        
        x_majorLocator = MultipleLocator(2)
        x_minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(5)
        y_minorLocator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        ax.set_title(self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt])
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'mag_num_hist_centarcsec.pdf')
        fig.savefig(plot_out_dir + 'mag_num_hist_centarcsec.png', dpi=200)
        
        plt.close(fig)
        
    
    def align_rms_resid_detections_plot(
        self,
        cent_arcsecond=False,
        combo_plot_runs=[0,1],
        label_rad = 0.001,
        circle_size=0.063 / 1.5,
        x_label_offset = 0,
        y_label_offset = 6,
    ):
        cur_wd = os.getcwd()
        
        starlist_align_location = self.align_rms_dir + '/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'
        
        plot_out_dir = self.epoch_analysis_location + 'resid_detections_plots/'
        os.makedirs(plot_out_dir, exist_ok=True)
        
        # Read in STF tables
        orig_run1_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt,
        )
        
        stf_run1_lis_table = stf_rms_lis_reader(orig_run1_stf_file)
        stf_run2_lis_table = stf_rms_lis_reader(orig_run2_stf_file)
        
        # Read in align tables
        
        align_pos_table = align_pos_reader(
            align_root + '.pos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_orig_pos_table = align_orig_pos_reader(
            align_root + '.origpos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        rad_distance = np.hypot(
            align_pos_table[self.run2_file_name_str + '_x'],
            align_pos_table[self.run2_file_name_str + '_y'],
        )
        
        close_star_index = np.argmin(rad_distance)
        close_star_op_x = (align_orig_pos_table[self.run2_file_name_str + '_x'])[close_star_index]
        close_star_op_y = (align_orig_pos_table[self.run2_file_name_str + '_y'])[close_star_index]
        close_star_pos_x = (align_pos_table[self.run2_file_name_str + '_x'])[close_star_index]
        close_star_pos_y = (align_pos_table[self.run2_file_name_str + '_y'])[close_star_index]
        
        center_op_x = close_star_op_x - (close_star_pos_x/self.plate_scale)
        center_op_y = close_star_op_y - (close_star_pos_y/self.plate_scale)
        
        # Common detections
        det_both = np.logical_and(
            align_orig_pos_table[self.run1_file_name_str + '_x'] != -100000.0,
            align_orig_pos_table[self.run2_file_name_str + '_x'] != -100000.0,
        )
        
        detection_filter = np.where(det_both)
        
        common_detections = align_orig_pos_table[detection_filter]
        common_detections_pos = align_pos_table[detection_filter]
        
        # Run 1 detections only
        run1_detection_filter = np.where(
            np.logical_and(
                align_orig_pos_table[self.run1_file_name_str + '_x'] != -100000.0,
                align_orig_pos_table[self.run2_file_name_str + '_x'] == -100000.0,
            ),
        )
        
        run1_only_detections = align_orig_pos_table[run1_detection_filter]
        run1_only_detections_pos = align_pos_table[run1_detection_filter]
        
        # Run 2 detections only
        run2_detection_filter = np.where(
            np.logical_and(
                align_orig_pos_table[self.run1_file_name_str + '_x'] == -100000.0,
                align_orig_pos_table[self.run2_file_name_str + '_x'] != -100000.0,
            ),
        )
        
        run2_only_detections = align_orig_pos_table[run2_detection_filter]
        run2_only_detections_pos = align_pos_table[run2_detection_filter]
        
        # Plot detections on the residual image
        stf_res_versions = [
            f'{self.run1_file_name_str}_combo',
            f'{self.run2_file_name_str}_combo',
            f'{self.run1_file_name_str}_res',
            f'{self.run2_file_name_str}_res',
        ]
        
        cent_arcsec_suffix = ''
        if cent_arcsecond:
            cent_arcsec_suffix = '_cent_arcsec'
        
        for image_index, stf_res_version in enumerate(stf_res_versions):
            if image_index < 2 and image_index not in combo_plot_runs:
                continue
            
            fig, ax = plt.subplots(figsize=(7,7))
            
            ax.set_title(
                self.epoch_name + ' ' + self.filt_label_strs[self.epoch_filt]
            )
            
            ax.axis('off')
            
            # Set up file path and name for FITS file of background image
            
            fits_file = ''
            
            if image_index == 0:
                fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
                    self.run1_dr_location,
                    self.epoch_name,
                    self.epoch_filt,
                )
            elif image_index == 1:
                fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
                    self.run2_dr_location,
                    self.epoch_name,
                    self.epoch_filt,
                )
            elif image_index == 2:
                fits_file = '{0}/mag{1}_{2}_res.fits'.format(
                    self.run1_location,
                    self.epoch_name,
                    self.epoch_filt,
                )
            elif image_index == 3:
                fits_file = '{0}/mag{1}_{2}_res.fits'.format(
                    self.run2_location,
                    self.epoch_name,
                    self.epoch_filt,
                )
            
            # Read image from the FITS file
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(fits_file) as hdulist:
                image_data = hdulist[0].data
            
            # Default AO image scaling
            im_add = 0.
            im_floor = 100.
            im_ceil = 1.e6
            im_ceil = 1.e4
            im_ceil = 5.e3
            
            im_mult = 1.
            im_invert = 1.
            
            # Res File image scaling
            if image_index > 1:
                im_add = 0.
                im_floor = -100.
                im_ceil = 400.
            
                im_floor = -1.5e3
                im_ceil = 1.5e3
            
                im_mult = 1.
                im_invert = 1.
            
            # Put in image floor
            
            image_data[np.where(image_data <= im_floor)] = im_floor
            image_data[np.where(image_data >= im_ceil)] = im_ceil
            
            image_data = (image_data - im_floor)
            
            # print(np.min(image_data))
            # print(np.max(image_data))
            
            image_data *= im_mult
            
            # Display image
            im_cmap = plt.get_cmap('gray')
            ax.imshow(im_invert * np.sqrt(image_data),
                      cmap=im_cmap,
                      interpolation='nearest')
            ax.invert_yaxis()
            
            # Center on 1 arcsecond around Sgr A*
            arcsec_pixels = 1.0/self.plate_scale
            
            ax.set_xlim([center_op_x - 5.3/self.plate_scale,
                         center_op_x + 5.3/self.plate_scale])
            ax.set_ylim([center_op_y - 6.5/self.plate_scale,
                         center_op_y + 4.1/self.plate_scale])
            
            if cent_arcsecond:
                ax.set_xlim([center_op_x - 1.0/self.plate_scale,
                             center_op_x + 1.0/self.plate_scale])
                ax.set_ylim([center_op_y - 1.0/self.plate_scale,
                             center_op_y + 1.0/self.plate_scale])
            
            # # Plot out detections
            
            # common_stars = []
            
            # for star_index in range(len(common_detections)):
            #     x = common_detections[self.run2_file_name_str + '_x'][star_index]
            #     y = common_detections[self.run2_file_name_str + '_y'][star_index]
                
            #     r2d = np.hypot(
            #         common_detections_pos[self.run2_file_name_str + '_x'][star_index],
            #         common_detections_pos[self.run2_file_name_str + '_y'][star_index],
            #     )
                
            #     c = ax.add_artist(
            #         plt.Circle(
            #             (x,y),
            #             radius=(circle_size * 1./self.plate_scale),
            #             linestyle='-', edgecolor='greenyellow',   # , edgecolor='C0'
            #             label='Detections in both modes',
            #             linewidth=1.5, fill=False,
            #         ),
            #     )
            #     common_stars.append(c)
                
            #     if r2d > label_rad:
            #         continue
                
            #     star_name = common_detections['name'][star_index]
                
            #     ax.text(
            #         x_label_offset + x,
            #         y_label_offset + y,
            #         star_name,
            #         ha='center', va='bottom', size='small',
            #         bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
            #                     facecolor = 'white', alpha = 0.8)
            #     )  # .replace('_', '\_')
            
            # run1_stars = []
            
            # for star_index in range(len(run1_only_detections)):
            #     x = run1_only_detections[self.run1_file_name_str + '_x'][star_index]
            #     y = run1_only_detections[self.run1_file_name_str + '_y'][star_index]
                
            #     r2d = np.hypot(
            #         run1_only_detections[self.run1_file_name_str + '_x'][star_index],
            #         run1_only_detections[self.run1_file_name_str + '_y'][star_index],
            #     )
                
            #     c = ax.add_artist(
            #         plt.Circle(
            #             (x, y),
            #             radius=(circle_size * 1./self.plate_scale),
            #             linestyle='-', edgecolor='C0',   # , edgecolor='C0'
            #             label=f'{self.run1_name_str} Only',
            #             linewidth=1.5, fill=False,
            #         ),
            #     )
            #     run1_stars.append(c)
                
            #     if r2d > label_rad:
            #         continue
                
            #     star_name = run1_only_detections['name'][star_index]
                
            #     ax.text(
            #         x_label_offset + x,
            #         y_label_offset + y,
            #         star_name,
            #         ha='left', va='bottom', size='small',
            #         bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
            #                     facecolor = 'white', alpha = 1.0)
            #     )  # .replace('_', '\_')
            
            # run2_stars = []
            
            # for star_index in range(len(run2_only_detections)):
            #     x = run2_only_detections[self.run2_file_name_str + '_x'][star_index]
            #     y = run2_only_detections[self.run2_file_name_str + '_y'][star_index]
                
            #     r2d = np.hypot(
            #         run2_only_detections[self.run2_file_name_str + '_x'][star_index],
            #         run2_only_detections[self.run2_file_name_str + '_y'][star_index],
            #     )
                
            #     c = ax.add_artist(
            #         plt.Circle(
            #             (x, y),
            #             radius=(circle_size * 1./self.plate_scale),
            #             linestyle='-', edgecolor='C1',
            #             label=f'{self.run2_name_str} Only',
            #             linewidth=1.5, fill=False,
            #         ),
            #     )
            #     run2_stars.append(c)
                
            #     if r2d > label_rad:
            #         continue
                
            #     star_name = run2_only_detections['name'][star_index]
                
            #     ax.text(
            #         x_label_offset + x,
            #         y_label_offset + y,
            #         star_name,
            #         ha='left', va='bottom', size='small',
            #         bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
            #                     facecolor = 'white', alpha = 0.8)
            #     )  # .replace('_', '\_')
            
            # ax.plot(center_op_x, center_op_y, 'ko', ms=14)
            # ax.plot(center_op_x, center_op_y, 'wo', ms=8)
            # ax.plot(center_op_x, center_op_y, 'ko', ms=4)
            
            # ax.legend(
            #     handles=[
            #         run1_stars[0], run2_stars[0], common_stars[0]
            #     ],
            #     loc='upper right', fontsize='medium',
            # )
            
            # Save out and close figure
            fig.tight_layout()
            
            fig.savefig(plot_out_dir + 'resid_{0}{1}.pdf'.format(
                stf_res_version, cent_arcsec_suffix,
            ))
            fig.savefig(plot_out_dir + 'resid_{0}{1}.png'.format(
                stf_res_version, cent_arcsec_suffix,
            ), dpi=200)
            
            plt.close(fig)
    
    def stf_resid_stats(
        self,
        combo_files_use = [0, 1],
    ):
        """
        Determine and plot StarFinder resid stats
        ---
        
        Parameters
        ----------
        combo_files_use : list of int, default=[0, 1]
            List of length 2 which lists which index of combo to use for each
            StarFinder run comparison. For example, can be [0, 0] if using
            first run (index 0) for the combo frame data for both
            StarFinder comparisons.
        """
        cur_wd = os.getcwd()
        
        starlist_align_location = self.align_rms_dir + '/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'
        
        plot_out_dir = self.epoch_analysis_location + 'resid_stats_plots/'
        os.makedirs(plot_out_dir, exist_ok=True)
        
        # Read in STF tables
        orig_run1_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run1_location, self.epoch_name,
            self.epoch_filt,
        )
        orig_run2_stf_file = '{0}/mag{1}_{2}_rms.lis'.format(
            self.run2_location, self.epoch_name,
            self.epoch_filt,
        )
        
        stf_run1_lis_table = stf_rms_lis_reader(orig_run1_stf_file)
        stf_run2_lis_table = stf_rms_lis_reader(orig_run2_stf_file)
        
        # Determine Sgr A* location using the align output
        align_pos_table = align_pos_reader(
            align_root + '.pos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        align_orig_pos_table = align_orig_pos_reader(
            align_root + '.origpos',
            run1_file_name_str=self.run1_file_name_str,
            run2_file_name_str=self.run2_file_name_str,
        )
        
        rad_distance_run1 = np.hypot(
            align_pos_table[self.run1_file_name_str + '_x'],
            align_pos_table[self.run1_file_name_str + '_y'],
        )
        
        rad_distance_run2 = np.hypot(
            align_pos_table[self.run2_file_name_str + '_x'],
            align_pos_table[self.run2_file_name_str + '_y'],
        )
        
        center_op_x = [0,0]
        center_op_y = [0,0]
        
        close_star_index = np.argmin(rad_distance_run1)
        close_star_op_x = (align_orig_pos_table[self.run1_file_name_str + '_x'])[close_star_index]
        close_star_op_y = (align_orig_pos_table[self.run1_file_name_str + '_y'])[close_star_index]
        close_star_pos_x = (align_pos_table[self.run1_file_name_str + '_x'])[close_star_index]
        close_star_pos_y = (align_pos_table[self.run1_file_name_str + '_y'])[close_star_index]
        
        center_op_x[0] = close_star_op_x - (close_star_pos_x/self.plate_scale)
        center_op_y[0] = close_star_op_y - (close_star_pos_y/self.plate_scale)
        
        close_star_index = np.argmin(rad_distance_run2)
        close_star_op_x = (align_orig_pos_table[self.run2_file_name_str + '_x'])[close_star_index]
        close_star_op_y = (align_orig_pos_table[self.run2_file_name_str + '_y'])[close_star_index]
        close_star_pos_x = (align_pos_table[self.run2_file_name_str + '_x'])[close_star_index]
        close_star_pos_y = (align_pos_table[self.run2_file_name_str + '_y'])[close_star_index]
        
        center_op_x[1] = close_star_op_x - (close_star_pos_x/self.plate_scale)
        center_op_y[1] = close_star_op_y - (close_star_pos_y/self.plate_scale)
        
        # Read in original image(s) and sig(s) FITS files
        image_data_orig = [
            np.array([]),
            np.array([]),
        ]
        
        image_sig_data = [
            np.array([]),
            np.array([]),
        ]
        
        if 0 in combo_files_use:
            image_fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
                self.run1_dr_location,
                self.epoch_name,
                self.epoch_filt,
            )
            
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(image_fits_file) as hdulist:
                image_data_orig[0] = hdulist[0].data
            
            image_sig_fits_file = '{0}/combo/{1}/mag{1}_{2}_sig.fits'.format(
                self.run1_dr_location,
                self.epoch_name,
                self.epoch_filt,
            )
            
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(image_sig_fits_file) as hdulist:
                image_sig_data[0] = hdulist[0].data
            
        if 1 in combo_files_use:
            image_fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
                self.run2_dr_location,
                self.epoch_name,
                self.epoch_filt,
            )
            
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(image_fits_file) as hdulist:
                image_data_orig[1] = hdulist[0].data
            
            image_sig_fits_file = '{0}/combo/{1}/mag{1}_{2}_sig.fits'.format(
                self.run2_dr_location,
                self.epoch_name,
                self.epoch_filt,
            )
            
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(image_sig_fits_file) as hdulist:
                image_sig_data[1] = hdulist[0].data
        
        image_data = [
            image_data_orig[combo_files_use[0]],
            image_data_orig[combo_files_use[1]],
        ]
        
        # Construct arrays of x and y coordinates
        (run1_image_y_len, run1_image_x_len) = (image_data[0]).shape
        
        run1_image_x_coords = np.arange(0, run1_image_x_len,),
        run1_image_y_coords = np.arange(0, run1_image_y_len,),
        
        run1_image_x_array_coords, run1_image_y_array_coords, = np.meshgrid(
            run1_image_x_coords,
            run1_image_y_coords,
        )
        
        (run2_image_y_len, run2_image_x_len) = (image_data[0]).shape
        
        run2_image_x_coords = np.arange(0, run2_image_x_len,),
        run2_image_y_coords = np.arange(0, run2_image_y_len,),
        
        run2_image_x_array_coords, run2_image_y_array_coords, = np.meshgrid(
            run2_image_x_coords,
            run2_image_y_coords,
        )
        
        # Determine x and y difference from Sgr A* position
        run1_sgra_x_diff = run1_image_x_array_coords - center_op_x[0]
        run1_sgra_y_diff = run1_image_y_array_coords - center_op_y[0]
        
        run2_sgra_x_diff = run2_image_x_array_coords - center_op_x[1]
        run2_sgra_y_diff = run2_image_y_array_coords - center_op_y[1]
        
        # Make a cut where distance is less than cut radius
        cut_pixel_radius = 1 / self.plate_scale
        
        run1_cent_arcsec_cut = np.where(
            np.hypot(
                run1_sgra_x_diff, run1_sgra_y_diff,
            ) < cut_pixel_radius
        )
        
        run2_cent_arcsec_cut = np.where(
            np.hypot(
                run2_sgra_x_diff, run2_sgra_y_diff,
            ) < cut_pixel_radius
        )
        
        # Read in resid images' data
        resid_data_run1_orig = np.array([])
        resid_data_run2_orig = np.array([])
        
        stf_res_fits_file = '{0}/mag{1}_{2}_res.fits'.format(
            self.run1_location,
            self.epoch_name,
            self.epoch_filt,
        )
        warnings.simplefilter('ignore', UserWarning)
        with fits.open(stf_res_fits_file) as hdulist:
            resid_data_run1_orig = hdulist[0].data
        
        stf_res_fits_file = '{0}/mag{1}_{2}_res.fits'.format(
            self.run2_location,
            self.epoch_name,
            self.epoch_filt,
        )
        warnings.simplefilter('ignore', UserWarning)
        with fits.open(stf_res_fits_file) as hdulist:
            resid_data_run2_orig = hdulist[0].data
        
        # Draw a 97% cut in sig to determine where on image to perform this analysis
        image_data = [
            image_data_orig[combo_files_use[0]],
            image_data_orig[combo_files_use[1]],
        ]
        resid_data = [
            resid_data_run1_orig,
            resid_data_run2_orig,
        ]
        
        image_data_centarcsec = [
            image_data_orig[combo_files_use[0]],
            image_data_orig[combo_files_use[1]],
        ]
        resid_data_centarcsec = [
            resid_data_run1_orig,
            resid_data_run2_orig,
        ]
        
        for run_index in [0, 1]:
            sig_max = np.max(image_sig_data[combo_files_use[run_index]])
            
            sig_cut = np.where(
                image_sig_data[combo_files_use[run_index]] >= 0.97 * sig_max
            )
            
            # Implement sig cuts in image and resid data
            image_data[run_index] = (image_data[run_index])[sig_cut]
            
            resid_data[run_index] = (resid_data[run_index])[sig_cut]
            
            # Implement image mask to only do analysis where image data is positive
            image_mask = np.where(image_data[run_index] > 0)
            
            image_data[run_index] = (image_data[run_index])[image_mask]
            
            resid_data[run_index] = (resid_data[run_index])[image_mask]
        
        image_data_centarcsec = [
            image_data_orig[combo_files_use[0]],
            image_data_orig[combo_files_use[1]],
        ]
        resid_data_centarcsec = [
            resid_data_run1_orig,
            resid_data_run2_orig,
        ]
        
        # for run_index in [0, 1]:
        #     # First cut out central arcsecond
            
        
        # Calculate abs resids and FEUs
        resid_abs_run1 = np.abs(resid_data[0].flatten())
        resid_abs_run2 = np.abs(resid_data[1].flatten())
        
        resid_abs_feu_run1 = resid_abs_run1 / (image_data[0].flatten())
        resid_abs_feu_run2 = resid_abs_run2 / (image_data[1].flatten())
        
        # Save out median data
        out_table = Table(
            [
                [np.nanmedian(resid_abs_run1)],
                [np.nanmedian(resid_abs_run2)],
                [np.nanmedian(resid_abs_feu_run1)],
                [np.nanmedian(resid_abs_feu_run2)],
            ],
            names=(
                f'median_resid_abs_{self.run1_file_name_str}',
                f'median_resid_abs_{self.run2_file_name_str}',
                f'median_resid_abs_feu_{self.run1_file_name_str}',
                f'median_resid_abs_feu_{self.run2_file_name_str}',
            ),
        )
        
        out_table.write(
            plot_out_dir + 'median_resids.txt',
            format='ascii.fixed_width_two_line',
            overwrite=True,
        )
        out_table.write(
            plot_out_dir + 'median_resids.h5',
            format='hdf5', path='data', serialize_meta=True,
            overwrite=True,
        )
        
        # Draw histogram
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.linspace(0, 1000, num=51)
        
        ax.hist(
            resid_abs_run1, bins=hist_bins,
            histtype='step',
            color='C0', label=self.run1_name_str,
        )
        
        ax.hist(
            resid_abs_run2, bins=hist_bins,
            histtype='step',
            color='C1', label=self.run2_name_str,
        )
        
        ax.axvline(
            np.nanmedian(resid_abs_run1),
            color='C0', ls='--',
        )
        ax.axvline(
            np.nanmedian(resid_abs_run2),
            color='C1', ls='--',
        )
        
        ax.set_xlabel(r'$|\mathrm{data} - \mathrm{model}|$')
        ax.set_ylabel('Number of pixels')
        
        ax.set_xlim([0, 1000])
        
        ax.legend()
        
        x_majorLocator = MultipleLocator(200)
        x_minorLocator = MultipleLocator(50)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(2e4)
        y_minorLocator = MultipleLocator(5e3)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'resid_hist.pdf')
        fig.savefig(plot_out_dir + 'resid_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Draw histogram for FEU
        fig, ax = plt.subplots(figsize=(6,3))
        
        hist_bins = np.linspace(0, 5, num=51)
        # hist_bins = np.linspace(0, 10, num=80)
        # hist_bins = np.logspace(-5, 2, num=100)
        
        ax.hist(
            resid_abs_feu_run1,
            bins=hist_bins,
            histtype='step',
            color='C0', label=self.run1_name_str,
        )
        
        ax.hist(
            resid_abs_feu_run2,
            bins=hist_bins,
            histtype='step',
            color='C1', label=self.run2_name_str,
        )
        
        ax.axvline(
            np.nanmedian(resid_abs_feu_run1),
            color='C0', ls='--',
        )
        ax.axvline(
            np.nanmedian(resid_abs_feu_run2),
            color='C1', ls='--',
        )
        
        ax.set_xlabel(r'$|\mathrm{data} - \mathrm{model}|$ / $\mathrm{data}$')
        ax.set_ylabel('Number of pixels')
        
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        ax.set_xlim([0, 5])
        
        # ax.legend(loc='upper left')
        ax.legend()
        
        x_majorLocator = MultipleLocator(1)
        x_minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(x_majorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        
        y_majorLocator = MultipleLocator(5e4)
        y_minorLocator = MultipleLocator(1e4)
        ax.yaxis.set_major_locator(y_majorLocator)
        ax.yaxis.set_minor_locator(y_minorLocator)
        
        fig.tight_layout()
        
        fig.savefig(plot_out_dir + 'resid_feu_hist.pdf')
        fig.savefig(plot_out_dir + 'resid_feu_hist.png', dpi=200)
        
        plt.close(fig)
        
        # Repeat, but just for central arcsecond
        
    ## Generating the master table.
        
    def generate_master_table_stats(self):
        
        ## directory to read table 
        
        stats_table_dir = self.epoch_analysis_location + 'align_rms_stats/' + 'overall_stats.txt'
        stats_table = pd.read_table(stats_table_dir, sep= '\s+', skiprows= [1])
    
        # print(stats_table.loc[1,'run_name'])
        
        sub_master_table_stats = pd.DataFrame()
        
        ## Copying data quality table values to the sub_master_table_stats
            # Bolean masking the epochs quality table to only have the current epochs info
            
        epochs_quality_table_mask= compare_epoch.epochs_quality_table[compare_epoch.epochs_quality_table['epoch'] == self.epoch_name]
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
        print(compare_epoch.master_table_stats)
