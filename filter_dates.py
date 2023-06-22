import bz2
from datetime import datetime
import pandas as pd
from lucupy.minimodel import Site
import os

start_date = datetime(2018, 1, 31, 12, 0)
end_date = datetime(2019, 8, 2, 12, 0)


def filter_dates(site: Site, input_file_name: str, output_file_name: str) -> None:
    with bz2.BZ2File(input_file_name, 'rb') as input_file:
        input_data = pd.read_pickle(input_file)
        input_data = input_data.drop(columns={
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

        input_data['Time_Stamp_UTC'] = pd.to_datetime(input_data['Time_Stamp_UTC'])
        filtered_df = input_data[(input_data['Time_Stamp_UTC'] >= start_date) & (input_data['Time_Stamp_UTC'] <= end_date)]
        filtered_df.to_pickle(output_file_name, compression='bz2')


def main():
    filter_dates(Site.GN, os.path.join('data', 'gn_wfs_filled_final_MEDIAN600s.pickle.bz2'), os.path.join('data', 'gn_filtered.pickle.bz2'))
    filter_dates(Site.GS, os.path.join('data', 'gs_wfs_filled_final_MEDIAN600s.pickle.bz2'), os.path.join('data', 'gs_filtered.pickle.bz2'))


if __name__ == '__main__':
    main()
