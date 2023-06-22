# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
import pandas as pd
from lucupy.minimodel import Site

from definitions import *


def filter_dates(site: Site, input_file_name: str, output_file_name: str) -> None:
    """
    Filters the data for the specified site, as given in the input_file_name.
    Data with dates outside the start_date and end_date are dropped, as well are unnecessary columns that we do
    not need in our final calculations. This reduces the size of the pandas dataframes and files significantly so
    that only essential data are kept, which allows env_processor.py to run much more quickly.
    """
    with bz2.BZ2File(input_file_name, 'rb') as input_file:
        df = pd.read_pickle(input_file)
        df = df.drop(columns={
            'Night', f'{site.name}_Sun_Elevation', 'cc_requested', 'iq_requested',
            'Airmass_QAP', 'Filter', 'Instrument', 'Object', 'Sidereal', 'Filename_QAP', 'Types',
            'adaptive_optics', 'calibration_program', 'cass_rotator_pa', 'data_label', 'dec',
            'engineering', 'exposure_time', 'filename', 'local_time', 'object', 'observation_class',
            'observation_id', 'observation_type', 'program_id', 'ra', 'science_verification',
            'types', 'wavefront_sensor', 'RA', 'DEC', 'Azimuth', 'Elevation', 'Airmass', 'Filename_FITS',
            'IQ_OIWFS', 'IQ_P2WFS', 'IQ_OIWFS_MEDIAN', 'IQ_P2WFS_MEDIAN', 'QAP_IQ', 'QAP_IQ_WFS_scaled',
            'Ratio_OIWFS', 'Ratio_P2WFS', 'QAP_IQ_WFS_scaled500', 'IQ_P2WFS_Zenith', 'IQ_OIWFS_Zenith',
            'IQ_P2WFS_MEDIAN_Zenith', 'QAP_IQ_WFS_scaled500_Zenith',
            'filter_name', 'instrument'})

        df[time_stamp_col] = pd.to_datetime(df[time_stamp_col])
        filtered_df = df[(df[time_stamp_col] >= first_date) & (df[time_stamp_col] <= last_date)]
        filtered_df.to_pickle(output_file_name, compression='bz2')


def main():
    sites = (
        Site.GN,
        Site.GS,
    )
    in_files = [os.path.join('data', f)
                for f in (
                    'gn_wfs_filled_final_MEDIAN600s.pickle.bz2',
                    'gs_wfs_filled_final_MEDIAN600s.pickle.bz2',
                )]
    out_files = [os.path.join('data', f)
                 for f in (
                     'gn_filtered.pickle.bz2',
                     'gs_filtered.pickle.bz2',
                 )]

    for site, in_file, out_file in zip(sites, in_files, out_files):
        filter_dates(site, in_file, out_file)


if __name__ == '__main__':
    main()
