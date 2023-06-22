# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bisect
import bz2
from enum import Enum
import logging
import os
from copy import copy
from datetime import timedelta, date, datetime
from math import ceil
from typing import Dict, List, Union, Type

from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz
from astropy.coordinates import Angle
from lucupy.helpers import lerp, lerp_enum, lerp_radians, round_minute
from lucupy.minimodel import Site, Variant, CloudCover, ImageQuality
from lucupy.sky import night_events

# In Hawaii:
# Earliest astronomical twilight (sunset): 18:56
# Latest astronomical twilight (sunrise): 5:40
# In UTC: 08:56 - 19:40 (all on one day)

# In Chile:
# Earliest astronomical twilight (sunset): 19:17
# Latest astronomical twilight (sunrise): 6:16
# In UTC:


_time_stamp = 'Time_Stamp_UTC'
_cc_band = 'cc_band'
_iq_band = 'iq_band'
_wind_speed = 'WindSpeed'
_wind_dir = 'WindDir'
_first_date = datetime(2018, 1, 31, 12, 0)
_last_date = datetime(2019, 8, 2, 12, 0)


def cc_band_to_float(data: str | float) -> float:
    """
    Convert a CC value from a str or float to a value in the enum.
    CC values in the pandas dataset are generally 100 times the value in the CloudCover enum.
    """
    # If it is not a number, return CCANY.
    if pd.isna(data):
        return 1.0

    # If it is a str, then it might be a set. If it is a set, eval it to get the set, and return the max.
    if type(data) == str and '{' in data:
        return max(eval(data)) / 100

    # Otherwise, it is a float, so return the value divided by 100.
    return float(data) / 100


def process_files(site: Site, input_file_name: str, output_file_name: str) -> None:
    """
    Process a pandas dataset for Gemini so that it has complete data for the night between twilights.
    For any missing values at the beginning or end of the night, insert ANY.
    For any missing internal values, linearly interpolate between the enum values.
    """
    print(f'*** Processing site: {site.name} ***')
    # Create the time grid for the site.
    start_time = Time(_first_date)
    end_time = Time(_last_date)
    time_grid = Time(np.arange(start_time.jd, end_time.jd + 1.0, (1.0 * u.day).value), format='jd')

    # Get all the twilights across the nights.
    _, _, _, twilight_evening_12, twilight_morning_12, _, _ = night_events(time_grid, site.location, site.timezone)

    # For convenience, length of a day.
    day = TimeDelta(1.0 * u.day)

    # Calculate the length of each night in the same way as the Collector.
    time_slot_length = TimeDelta(1.0 * u.min)
    time_slot_length_days = time_slot_length.to(u.day).value
    time_starts = round_minute(twilight_evening_12, up=True)
    time_ends = round_minute(twilight_morning_12, up=True)
    time_slots_per_night = ((time_ends.jd - time_starts.jd) / time_slot_length_days + 0.5).astype(int)

    # Create the times array, which is the number of time slots per night in jd format.
    times = [Time(np.linspace(start.jd, end.jd - time_slot_length_days, i), format='jd')
             for start, end, i in zip(time_starts, time_ends, time_slots_per_night)]

    # Convert to UTC.
    utc_times = [t.to_datetime(pytz.UTC) for t in times]

    with bz2.open(input_file_name) as input_file:
        # Read it in as a pandas dataframe.
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

        input_frame = pd.DataFrame(input_data)

        # Add timezone information to the _time_stamp column to get between to pass.
        input_frame[_time_stamp] = input_frame[_time_stamp].dt.tz_localize('UTC')

        # Create a row with worst values to fill in gaps.
        # We will have to copy and set the timestamp.
        empty_row = copy(input_frame.iloc[0])
        empty_row[_iq_band] = 1.0
        empty_row[_cc_band] = 1.0
        empty_row[_wind_speed] = 0.0
        empty_row[_wind_dir] = 0.0

        # Store the data by night.
        data_by_night = {}

        # Process each night.
        pd_minute = pd.Timedelta(minutes=1)
        pd_time_starts = pd.to_datetime(time_starts.isot, utc=True).to_numpy()
        pd_time_ends = pd.to_datetime(time_ends.isot, utc=True).to_numpy()
        for night_idx, pd_time_info in enumerate(zip(pd_time_starts, pd_time_ends)):
            pd_start_time, pd_end_time = pd_time_info

            # Night date
            curr_date = pd_start_time.date()

            pd_curr_time = pd_start_time

            # The rows for the night.
            night_rows = []

            # Get all the rows that fall between the twilights and sort the data by timestamp.
            night_input_frame = input_frame[input_frame[_time_stamp].between(pd_start_time,
                                                                             pd_end_time,
                                                                             inclusive='both')]
            night_input_frame.sort_values(by=_time_stamp)

            # SPECIAL CASE: NO INPUT FOR NIGHT.
            if night_input_frame.empty:
                print(f'No data for {curr_date}... adding data.')

                # Loop from pd_start_time to pd_end_time.
                while pd_curr_time < pd_end_time:
                    new_row = copy(empty_row)
                    new_row[_time_stamp] = pd_curr_time
                    night_rows.append(new_row)
                    pd_curr_time += pd_minute

                # Loop to next night.
                data_by_night[curr_date] = night_rows
                print(f'Time slots expected: {time_slots_per_night[night_idx]}, '
                      f'time slots filled: {len(data_by_night[curr_date])}')
                continue

            # *** REGULAR CASE: Some data for date. ***
            print(f'Adding data for {curr_date}...')
            prev_row = None

            # Fill in any missing data at beginning of night.
            pd_first_time_in_frame = night_input_frame.iloc[0][_time_stamp]
            while pd_curr_time < pd_first_time_in_frame:
                new_row = copy(empty_row)
                new_row[_time_stamp] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute
                prev_row = new_row

            # Iterate over the rows.
            for idx, curr_row in night_input_frame.iterrows():
                # Adjust the values in pd_row to be standardized and to eliminate nan.
                curr_row[_cc_band] = cc_band_to_float(curr_row[_cc_band])
                curr_row[_iq_band] = 1.0 if pd.isna(curr_row[_iq_band]) else curr_row[_iq_band] / 100
                if pd.isna(curr_row[_wind_dir]): curr_row[_wind_dir] = 0.0
                if pd.isna(curr_row[_wind_speed]): curr_row[_wind_speed] = 0.0

                # Get the timestamp for the current row and determine if there is missing data.
                pd_next_time = curr_row[_time_stamp]
                gaps = int((pd_next_time - pd_curr_time).total_seconds() / 60 - 1)

                # Fill in any gaps between the prev_row and the curr_row with linear interpolation.
                if gaps > 0:
                    print(f'Interpolating CloudCover between {prev_row[_cc_band]} and {curr_row[_cc_band]} for {gaps}.')
                    cc = lerp_enum(CloudCover, prev_row[_cc_band], curr_row[_cc_band], gaps)
                    print(f'Complete: {cc}')
                    print(f'Interpolating ImageQuality between {prev_row[_cc_band]} and {curr_row[_cc_band]} for {gaps}.')
                    iq = lerp_enum(ImageQuality, prev_row[_iq_band], curr_row[_iq_band], gaps)
                    print(f'Complete: {iq}')
                    wind_dir = lerp_radians(prev_row[_wind_dir], curr_row[_wind_dir], gaps)
                    wind_speed = lerp(prev_row[_wind_speed], curr_row[_wind_speed], gaps)

                    ii = 0
                    while pd_curr_time < pd_next_time:
                        new_row = copy(empty_row)
                        new_row[_time_stamp] = pd_curr_time
                        new_row[_cc_band] = cc[ii]
                        new_row[_iq_band] = iq[ii]
                        new_row[wind_dir] = wind_dir[ii]
                        new_row[_wind_speed] = wind_speed[ii]
                        night_rows.append(new_row)
                        ii += 1
                        pd_curr_time += pd_minute

                # Add the current row and designate it to the prev_row.
                night_rows.append(curr_row)
                prev_row = curr_row

            # Advance to the next minute and fill in any missing data.
            pd_curr_time += pd_minute

            while pd_curr_time < pd_end_time:
                new_row = copy(empty_row)
                new_row[_time_stamp] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute

            data_by_night[curr_date] = night_rows
            print(f'Time slots expected: {time_slots_per_night[night_idx]}, '
                  f'time slots filled: {len(data_by_night[curr_date])}')

        print('Done.')


def main():
    sites = [Site.GN, Site.GS]
    in_files = [os.path.join('data', f) for f in ('gn_wfs_filled_final_MEDIAN600s.pickle.bz2',
                                                  'gs_wfs_filled_final_MEDIAN600s.pickle.bz2',)]
    out_files = [os.path.join('data', f) for f in ('gn_weather_data.bz2',
                                                   'gs_weather_data.bz2',)]

    for site, in_file, out_file in zip(sites, in_files, out_files):
        process_files(site, in_file, out_file)


if __name__ == '__main__':
    main()
