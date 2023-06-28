# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
from copy import copy
from tqdm import tqdm

from astropy.time import Time
from astropy import units as u
import numpy as np
import pandas as pd
from lucupy.helpers import lerp, lerp_degrees, lerp_enum, round_minute
from lucupy.minimodel import Site, CloudCover, ImageQuality
from lucupy.sky import night_events

from definitions import *


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
    start_time = Time(first_date)
    end_time = Time(last_date)
    time_grid = Time(np.arange(start_time.jd, end_time.jd + 1.0, (1.0 * u.day).value), format='jd')

    # Get all the twilights across the nights.
    _, _, _, twilight_evening_12, twilight_morning_12, _, _ = night_events(time_grid, site.location, site.timezone)

    # Calculate the length of each night in the same way as the Collector.
    time_starts = round_minute(twilight_evening_12, up=True)
    time_ends = round_minute(twilight_morning_12, up=True)

    with bz2.BZ2File(input_file_name, 'rb') as input_file:
        df = pd.read_pickle(input_file)

        # Add timezone information to the time_stamp_col to get between to pass.
        df[time_stamp_col] = df[time_stamp_col].dt.tz_localize('UTC')

        # Create a row with worst values to fill in gaps.
        # We will have to copy and set the timestamp.
        print(df.iloc[0])
        empty_row = copy(df.iloc[0])
        empty_row[iq_band_col] = 1.0
        empty_row[cc_band_col] = 1.0
        empty_row[wind_speed_col] = 0.0
        empty_row[wind_dir_col] = 0.0

        # Store the data by night.
        data_by_night = {}

        # Process each night.
        pd_minute = pd.Timedelta(minutes=1)
        pd_time_starts = pd.to_datetime(time_starts.isot, utc=True).to_numpy()
        pd_time_ends = pd.to_datetime(time_ends.isot, utc=True).to_numpy()
        for night_idx, pd_time_info in tqdm(enumerate(zip(pd_time_starts, pd_time_ends))):
            pd_start_time, pd_end_time = pd_time_info

            # Night date
            curr_date = pd_start_time.date()

            pd_curr_time = pd_start_time

            # The rows for the night.
            night_rows = []

            # Get all the rows that fall between the twilights and sort the data by timestamp.
            night_df = df[df[time_stamp_col].between(pd_start_time,
                                                     pd_end_time,
                                                     inclusive='both')]
            night_df.sort_values(by=time_stamp_col)

            # SPECIAL CASE: NO INPUT FOR NIGHT.
            if night_df.empty:
                # print(f'No data for {curr_date}... adding data.')

                # Loop from pd_start_time to pd_end_time.
                while pd_curr_time < pd_end_time:
                    new_row = copy(empty_row)
                    new_row[time_stamp_col] = pd_curr_time
                    night_rows.append(new_row)
                    pd_curr_time += pd_minute

                # Loop to next night.
                data_by_night[curr_date] = night_rows
                # print(f'Time slots expected: {time_slots_per_night[night_idx]}, '
                #       f'time slots filled: {len(data_by_night[curr_date])}')
                continue

            # *** REGULAR CASE: Some data for date. ***
            # print(f'\n\nAdding data for {curr_date}...')
            prev_row = None

            # Fill in any missing data at beginning of night.
            start_gaps = 0
            pd_first_time_in_frame = night_df.iloc[0][time_stamp_col]
            while pd_curr_time < pd_first_time_in_frame:
                start_gaps += 1
                new_row = copy(empty_row)
                new_row[time_stamp_col] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute
                prev_row = new_row

            # Iterate over the rows.
            for idx, curr_row in night_df.iterrows():
                # Adjust the values in pd_row to be standardized and to eliminate nan.
                curr_row[cc_band_col] = cc_band_to_float(curr_row[cc_band_col])
                curr_row[iq_band_col] = 1.0 if pd.isna(curr_row[iq_band_col]) else curr_row[iq_band_col] / 100
                if pd.isna(curr_row[wind_dir_col]):
                    curr_row[wind_dir_col] = 0.0
                if pd.isna(curr_row[wind_speed_col]):
                    curr_row[wind_speed_col] = 0.0

                # Get the timestamp for the current row and determine if there is missing data.
                pd_next_time = curr_row[time_stamp_col]
                gaps = int((pd_next_time - pd_curr_time).total_seconds() / 60)

                # Fill in any gaps between the prev_row and the curr_row with linear interpolation.
                if gaps > 0:
                    cc = lerp_enum(CloudCover, prev_row[cc_band_col], curr_row[cc_band_col], gaps)
                    iq = lerp_enum(ImageQuality, prev_row[iq_band_col], curr_row[iq_band_col], gaps)
                    wind_dir = lerp_degrees(prev_row[wind_dir_col], curr_row[wind_dir_col], gaps)
                    wind_speed = lerp(prev_row[wind_speed_col], curr_row[wind_speed_col], gaps)

                    ii = 0
                    while ii < gaps:
                        new_row = copy(empty_row)
                        new_row[time_stamp_col] = pd_curr_time
                        new_row[cc_band_col] = cc[ii]
                        new_row[iq_band_col] = iq[ii]
                        new_row[wind_dir_col] = wind_dir[ii]
                        new_row[wind_speed_col] = wind_speed[ii]
                        night_rows.append(new_row)
                        ii += 1
                        pd_curr_time += pd_minute

                # Add the current row and designate it to the prev_row.
                print(curr_row)
                print(curr_row[wind_speed_col])
                night_rows.append(curr_row)
                prev_row = curr_row
                pd_curr_time = curr_row[time_stamp_col] + pd_minute

            end_gaps = 0
            while pd_curr_time < pd_end_time:
                end_gaps += 1
                new_row = copy(empty_row)
                new_row[time_stamp_col] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute

            data_by_night[curr_date] = night_rows

        # Flatten the data back down to a table.
        flattened_data = []
        for rows in data_by_night.values():
            flattened_data.extend(rows)

        # Convert back to a pandas dataframe and store.
        modified_df = pd.DataFrame(flattened_data)
        modified_df.to_pickle(output_file_name, compression='bz2')

        print('Done.')


def main():
    sites = (
        Site.GN,
        Site.GS,
    )
    in_files = [os.path.join('data', f) for f in (
        'gn_filtered.pickle.bz2',
        'gs_filtered.pickle.bz2',
    )]
    out_files = [os.path.join('data', f) for f in (
        'gn_weather_data.pickle.bz2',
        'gs_weather_data.pickle.bz2',
    )]

    for site, in_file, out_file in zip(sites, in_files, out_files):
        process_files(site, in_file, out_file)


if __name__ == '__main__':
    main()
