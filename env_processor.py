# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
from copy import copy
from enum import Enum
import logging
from typing import List, Optional, Type

from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np
import pandas as pd
from lucupy.helpers import lerp, lerp_degrees, lerp_enum, round_minute
from lucupy.minimodel import Site, CloudCover, ImageQuality
from lucupy.sky import night_events

from definitions import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

MINUTE: TimeDelta = TimeDelta(60, format='sec')


def val_to_float(data: str | float) -> Optional[float]:
    """
    Convert a CC value from a str or float to a value in the enum.
    CC values in the pandas dataset are generally 100 times the value in the CloudCover enum.
    """
    # If it is not a number, i.e. nan or None, return None.
    if pd.isna(data):
        return None

    # If it is a str, then it might be a set. If it is a set, eval it to get the set, and return the max.
    if type(data) == str and "{" in data:
        s = eval(data)
        value = max(s) / 100
        return value

    # Otherwise, it is a float, so return the value divided by 100.
    return float(data) / 100


def clean_gaps(night_rows: List, column_name: str, enum: Optional[Type[Enum]] = None) -> None:
    """
    Given a set of rows and a column, collect all the na entries from the rows in that column.
    Then perform a discrete linear interpolation to fill them in.
    """
    if pd.isna(night_rows[0][column_name]):
        night_rows[0][column_name] = 1.0
    if pd.isna(night_rows[-1][column_name]):
        night_rows[-1][column_name] = 1.0

    prev_row = None
    row_block = []
    for curr_row in night_rows:
        if prev_row is None:
            prev_row = curr_row
            continue

        # If we have an na value, append to the row block for processing.
        # Otherwise, process the row block and fill with the linear interpolation.
        if pd.isna(curr_row[column_name]):
            row_block.append(curr_row)
        else:
            if len(row_block) > 0:
                # Perform the linear interpolation and fill in the values.
                if enum is None:
                    lerp_entries = lerp(prev_row[column_name], curr_row[column_name], len(row_block))
                else:
                    lerp_entries = lerp_enum(enum, prev_row[column_name], curr_row[column_name], len(row_block))
            
                for row, value in zip(row_block, lerp_entries):
                    row[column_name] = value
                row_block = []
            prev_row = curr_row


def process_files(site: Site, input_file_name: str, output_file_name: str) -> None:
    """
    Process a pandas dataset for Gemini so that it has complete data for the night between twilights.
    For any missing values at the beginning or end of the night, insert ANY.
    For any missing internal values, linearly interpolate between the enum values.
    """
    logger.info(f"*** Processing site: {site.name} ***")
    # Create the time grid for the site.
    start_time = Time(first_date)
    end_time = Time(last_date)
    time_grid = Time(np.arange(start_time.jd, end_time.jd + 1.0, (1.0 * u.day).value), format="jd")

    # Get all the twilights across the nights.
    _, _, _, twilight_evening_12, twilight_morning_12, _, _ = night_events(time_grid, site.location, site.timezone)

    # Calculate the length of each night in the same way as the Collector.
    time_starts = round_minute(twilight_evening_12, up=True)
    time_ends = round_minute(twilight_morning_12, up=True)

    with bz2.BZ2File(input_file_name, "rb") as input_file:
        df = pd.read_pickle(input_file)

        # Add timezone information to the time_stamp_col to get between to pass.
        df[time_stamp_col] = df[time_stamp_col].dt.tz_localize("UTC")

        # Create an empty row to fill in intermediate gaps.
        # We will have to copy and set the timestamp.
        # There are many rows where CC and / or IQ are nan or None. To handle these cases, we will interpolate over
        # the data after inserting all the empty rows.
        empty_row = copy(df.iloc[0])
        empty_row[iq_band_col] = None
        empty_row[cc_band_col] = None
        empty_row[wind_speed_col] = 0.0
        empty_row[wind_dir_col] = 0.0

        # Create a "bad" row with the worst conditions to fill in missing gaps at beginning / ends of nights or
        # rows for entirely missing days.
        bad_row = copy(empty_row)
        bad_row[iq_band_col] = 1.0
        bad_row[cc_band_col] = 1.0

        # Store the data by night.
        data_by_night = {}

        # Process each night.
        pd_minute = pd.Timedelta(minutes=1)
        pd_time_starts = pd.to_datetime(time_starts.isot, utc=True).to_numpy()
        pd_time_ends = pd.to_datetime(time_ends.isot, utc=True).to_numpy()
        assert(len(pd_time_ends) == len(pd_time_ends))

        time_slot_length = TimeDelta(1.0 * u.min)
        time_slot_length_days = time_slot_length.to(u.day).value

        # TODO: Does this really calculate the correct number of time slots? It seems we may sometimes be one short.
        # TODO: This will impact the NightEvents in the Scheduler.
        time_slots_per_night = ((time_ends.jd - time_starts.jd) / time_slot_length_days + 0.5).astype(int)

        for night_idx, pd_time_info in enumerate(zip(pd_time_starts, pd_time_ends)):
            pd_start_time, pd_end_time = pd_time_info

            # Night date
            curr_date = pd_start_time.date()

            pd_curr_time = pd_start_time

            # The rows for the night.
            night_rows = []

            # Get all the rows that fall between the twilights and sort the data by timestamp.
            night_df = df[
                df[time_stamp_col].between(pd_start_time, pd_end_time, inclusive="both")
            ]
            night_df.sort_values(by=time_stamp_col)

            # SPECIAL CASE: NO INPUT FOR NIGHT.
            if night_df.empty:
                # Loop from pd_start_time to pd_end_time, inserting the worst conditions.
                # TODO: Check to see what we should use for wind speed and direction.
                while pd_curr_time < pd_end_time:
                    new_row = copy(bad_row)
                    new_row[time_stamp_col] = pd_curr_time
                    night_rows.append(new_row)
                    pd_curr_time += pd_minute

                # Loop to next night.
                data_by_night[curr_date] = night_rows

                if time_slots_per_night[night_idx] != len(data_by_night[curr_date]):
                    logger.error(f'Site: {site}, Night: {curr_row}: Expected: {time_slots_per_night[night_idx]} '
                                 f'Actual: {len(data_by_night[curr_date])} (fully generated)')

                continue

            # *** REGULAR CASE: Some data for date. ***
            prev_row = None

            # Fill in any missing data at beginning of night.
            start_gaps = 0
            pd_first_time_in_frame = night_df.iloc[0][time_stamp_col]
            while pd_curr_time < pd_first_time_in_frame:
                start_gaps += 1
                new_row = copy(bad_row)
                new_row[time_stamp_col] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute
                prev_row = new_row

            # Iterate over the rows.
            for idx, curr_row in night_df.iterrows():
                curr_row[cc_band_col] = val_to_float(curr_row[cc_band_col])
                curr_row[iq_band_col] = val_to_float(curr_row[iq_band_col])
                if pd.isna(curr_row[wind_dir_col]) or pd.isna(curr_row[wind_speed_col]):
                    logger.warning(f'Missing wind data for {site.name} {curr_row[time_stamp_col]}: '
                                   f'speed={curr_row[wind_speed_col]}, dir={curr_row[wind_dir_col]}')

                # Check for jumps. These are in the data sets and not caused by the scripts.
                if prev_row is not None:
                    cc_prev = prev_row[cc_band_col]
                    cc_curr = curr_row[cc_band_col]
                    # if prev_row is not None and abs(curr_row[cc_band_col] - prev_row[cc_band_col]) >= 0.5:
                    if cc_curr is not None and cc_prev is not None and abs(cc_curr - cc_prev) >= 0.5:
                        logger.warning(f'CC jump for {site.name} {curr_row[time_stamp_col]}: {cc_prev} to {cc_curr}')

                    iq_prev = prev_row[iq_band_col]
                    iq_curr = curr_row[iq_band_col]
                    if iq_curr is not None and iq_prev is not None and abs(iq_curr - iq_prev) in {0.3, 0.65}:
                        logger.warning(f'IQ jump for {site.name} {curr_row[time_stamp_col]}: {cc_prev} to {cc_curr}')

                # Get the timestamp for the current row and determine if there is missing data.
                pd_next_time = curr_row[time_stamp_col]
                gaps = int((pd_next_time - pd_curr_time).total_seconds() / 60)

                # Fill in any gaps in wind data between the prev_row and the curr_row with linear interpolation.
                # We will take another pass through the data to fill in CC and IQ entries due to the abundance
                # of nan and None entries.
                if gaps > 0:
                    wind_dir = lerp_degrees(prev_row[wind_dir_col], curr_row[wind_dir_col], gaps)
                    wind_speed = lerp(prev_row[wind_speed_col], curr_row[wind_speed_col], gaps)

                    ii = 0
                    while ii < gaps:
                        new_row = copy(empty_row)
                        new_row[time_stamp_col] = pd_curr_time
                        new_row[wind_dir_col] = wind_dir[ii]
                        new_row[wind_speed_col] = wind_speed[ii]
                        night_rows.append(new_row)
                        ii += 1
                        pd_curr_time += pd_minute

                # Add the current row and designate it to the prev_row.
                # print(curr_row)
                night_rows.append(curr_row)
                prev_row = curr_row
                pd_curr_time = curr_row[time_stamp_col] + pd_minute

            end_gaps = 0
            while pd_curr_time < pd_end_time:
                end_gaps += 1
                new_row = copy(bad_row)
                new_row[time_stamp_col] = pd_curr_time
                night_rows.append(new_row)
                pd_curr_time += pd_minute

            # Clean up the gaps for IQ and CC.
            clean_gaps(night_rows, iq_band_col, ImageQuality)
            clean_gaps(night_rows, cc_band_col, CloudCover)
            clean_gaps(night_rows, wind_speed_col)
            clean_gaps(night_rows, wind_dir_col)

            # Now iterate over the night blocks to make sure every entry has valid data and that the entries are
            # spaced one minute apart.
            # TODO: We are having issues where the entries are spaced correctly, but one more row than expected is made.
            prev_row = None
            for curr_row in night_rows:
                if prev_row is not None:
                    difference = curr_row[time_stamp_col] - prev_row[time_stamp_col]
                    assert(difference == MINUTE)
                assert(not pd.isna(curr_row[iq_band_col]))
                assert(not pd.isna(curr_row[cc_band_col]))
                assert(not pd.isna(curr_row[wind_dir_col]))
                assert(not pd.isna(curr_row[wind_speed_col]))
                prev_row = curr_row

            data_by_night[curr_date] = night_rows

            time_slot_difference = time_slots_per_night[night_idx] - len(night_rows)
            if time_slot_difference:
                # Get the expected values and convert to datetime to standardize.
                first_expected = time_starts[night_idx]
                first_expected.format = 'datetime'
                last_expected = time_ends[night_idx]
                last_expected.format = 'datetime'
                expected_slots = int((last_expected.datetime - first_expected.datetime).total_seconds() / 60 + 1)

                pd_first_actual = night_rows[0][time_stamp_col].to_pydatetime().replace(tzinfo=None)
                pd_last_actual = night_rows[-1][time_stamp_col].to_pydatetime().replace(tzinfo=None)
                first_actual = Time(pd_first_actual, scale='utc')
                last_actual = Time(pd_last_actual, scale='utc')
                actual_slots = int((last_actual.datetime - first_actual.datetime).total_seconds() / 60 + 1)

                logger.warning(f'Site: {site}, Night: {curr_date}: Expected: {time_slots_per_night[night_idx]} '
                               f'Actual: {len(data_by_night[curr_date])} (input), difference={time_slot_difference}\n'
                               f'\tFirst expected: {first_expected}, last expected: {last_expected}\n'
                               f'\tFirst actual:   {first_actual}, last actual:   {last_actual}\n'
                               f'\tCalc expected time slots: {expected_slots}, Calc actual time slots: {actual_slots}')

        # Flatten the data back down to a table.
        flattened_data = []
        for rows in data_by_night.values():
            flattened_data.extend(rows)

        # Convert back to a pandas dataframe and store.
        modified_df = pd.DataFrame(flattened_data)
        modified_df.to_pickle(output_file_name, compression="bz2")

        logger.info(f"*** Done processing site: {site.name} ***")


def main():
    sites = (
        Site.GN,
        Site.GS,
    )
    in_files = [
        os.path.join("data", f)
        for f in ("gn_filtered.pickle.bz2", "gs_filtered.pickle.bz2",)
    ]
    out_files = [
        os.path.join("data", f)
        for f in ("gn_weather_data.pickle.bz2", "gs_weather_data.pickle.bz2",)
    ]

    for site, in_file, out_file in zip(sites, in_files, out_files):
        process_files(site, in_file, out_file)


if __name__ == "__main__":
    main()
