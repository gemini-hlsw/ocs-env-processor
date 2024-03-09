# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

# Preprocess data to filter out illegal dates, which will greatly reduce the size of the data set.

import bz2
import pandas as pd
from lucupy.minimodel import Site

from definitions import first_date, last_date, local_time_stamp_col, utc_time_stamp_col, ROOT_DIR


def filter_dates(site: Site, input_file_name: str, output_file_name: str) -> None:
    """
    Filters the data for the specified site, as given in the input_file_name.
    Data with dates outside the start_date and end_date are dropped, as well are unnecessary columns that we do
    not need in our final calculations. This reduces the size of the pandas dataframes and files significantly so
    that only essential data are kept, which allows env_processor.py to run much more quickly.
    """
    with bz2.BZ2File(input_file_name, 'rb') as input_file:
        df = pd.read_pickle(input_file)

        # print('+++ Column information +++')
        # for column_name, column_type in df.dtypes.items():
        #     print(f"Column: {column_name}, Type: {column_type}")

        # Note that Time_Stamp is not reliable: sometimes it is timezone-aware and other times it is not.
        # If local_time is nan, then Time_Stamp is timezone-aware.
        # If local_time is defined, then Time_Stampe is not timesonze-aware.
        # print('\n\n+++ Date information +++')
        # for idx, row in df.iterrows():
        #     print(f'{idx}   {row["Time_Stamp"]}   {row["local_time"]}   {row[utc_time_stamp_col]}')

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

        # Make the timestamp column tz-aware to UTC and add a local time stamp column that is timezone-aware..
        df[utc_time_stamp_col] = pd.to_datetime(df[utc_time_stamp_col])
        df[utc_time_stamp_col] = df[utc_time_stamp_col].dt.tz_localize('UTC')
        df[local_time_stamp_col] = df[utc_time_stamp_col].dt.tz_convert(site.timezone)

        filtered_df = df[(df[local_time_stamp_col] >= first_date[site]) & (df[local_time_stamp_col] <= last_date[site])]
        filtered_df.to_pickle(output_file_name, compression='bz2')


def main():
    gn_data = (
        Site.GN,
        ROOT_DIR / 'data' / 'gn_wfs_filled_final.pickle.bz2',
        ROOT_DIR / 'data' / 'gn_filtered.pickle.bz2'
    )
    gs_data = (
        Site.GS,
        ROOT_DIR / 'data' / 'gs_wfs_filled_final.pickle.bz2',
        ROOT_DIR / 'data' / 'gs_filtered.pickle.bz2'
    )

    for site, in_file, out_file in [gn_data, gs_data]:
        filter_dates(site, in_file, out_file)


if __name__ == '__main__':
    main()
