# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
from copy import copy
from enum import Enum
from typing import Callable, List, Optional, Type

from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from lucupy.helpers import lerp, lerp_degrees, lerp_enum, round_minute
from lucupy.minimodel import Site, CloudCover, ImageQuality
from lucupy.sky import night_events

from definitions import *

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


def clean_gaps(night_rows: List,
               column_name: str,
               description: str,
               default_value: float,
               lerper: Callable[[float, float, int], npt.NDArray[float]]) -> None:
    """
    Given a set of rows and a column, collect all the na entries from the rows in that column.
    Then perform a discrete linear interpolation to fill them in.
    """
    if pd.isna(night_rows[0][column_name]):
        night_rows[0][column_name] = default_value
    if pd.isna(night_rows[-1][column_name]):
        night_rows[-1][column_name] = default_value

    prev_row = None
    row_block = []
    for curr_row in night_rows:
        if prev_row is None:
            prev_row = curr_row
            continue

        # If we have a na value, append to the row block for processing.
        # Otherwise, process the row block and fill with the linear interpolation.
        if pd.isna(curr_row[column_name]):
            row_block.append(curr_row)
        else:
            if len(row_block) > 0:
                # Perform the linear interpolation and fill in the values.
                lerp_entries = lerper(prev_row[column_name], curr_row[column_name], len(row_block))
                print(f'\tlerp on {description} from {prev_row[column_name]} to {curr_row[column_name]} for '
                      f'{len(row_block)}: {list(lerp_entries)}')
                for row, value in zip(row_block, lerp_entries):
                    row[column_name] = value
                row_block = []
            prev_row = curr_row


def enum_gaps(enum: Type[Enum]) -> Callable[[float, float, int], npt.NDArray[float]]:
    """
    Used to create a lerper over a binned enum value, i.e. CloudCover or ImagqQuality.
    """
    def f(a: float, b: float, g: int) -> npt.NDArray[float]:
        return lerp_enum(enum, a, b, g)
    return f


def continuous_gaps(lerp_func) -> Callable[[float, float, int], npt.NDArray[float]]:
    """
    Used to create a continuous lerper, i.e. wind speed or wind direction.
    """
    def f(a: float, b: float, g: int) -> npt.NDArray[float]:
        return lerp_func(a, b, g)
    return f


def process_files(site: Site, input_file_name: str, output_file_name: str) -> None:
    """
    Process a pandas dataset for Gemini so that it has complete data for the night between twilights.
    For any missing values at the beginning or end of the night, insert ANY.
    For any missing internal values, linearly interpolate between the enum values.
    """
    print(f"*** Processing site: {site.name} ***")

    # Create the time grid for the site.
    tg_time_start = Time(first_date)
    tg_time_end = Time(last_date)
    time_grid = Time(np.arange(tg_time_start.jd, tg_time_end.jd + 1.0, (1.0 * u.day).value), format="jd")

    # Calculate the number of time slots per night as the collector does it. This is indexed by night_idx.
    # Variables prefixed with c_ for collector.
    c_minute = TimeDelta(1.0 * u.min)
    _, _, _, twilight_evening_12, twilight_morning_12, _, _ = night_events(time_grid, site.location, site.timezone)
    c_time_starts = round_minute(twilight_evening_12, up=True)
    c_time_ends = round_minute(twilight_morning_12, up=True)
    c_time_slot_length_days = c_minute.to(u.day).value
    c_time_slots_per_night = ((c_time_ends.jd - c_time_starts.jd) / c_time_slot_length_days + 0.5).astype(int)

    # Calculate the length of each night in pandas format.
    # Variable prefixed with pd_ for pandas.
    pd_minute = pd.Timedelta(minutes=1)
    pd_time_starts = pd.to_datetime(c_time_starts.isot, utc=True).to_numpy()
    pd_time_ends = pd.to_datetime(c_time_ends.isot, utc=True).to_numpy()
    assert len(pd_time_starts) == len(pd_time_ends)

    with bz2.BZ2File(input_file_name, "rb") as input_file:
        df = pd.read_pickle(input_file)

        # Convert raw values
        df.raw_cc = df.raw_cc / 100
        df.raw_iq = df.raw_iq / 100

        # Add timezone information to the time_stamp_col to get between to pass.
        df[time_stamp_col] = df[time_stamp_col].dt.tz_localize("UTC")

        # Create an empty row to fill in intermediate gaps.
        # We will have to copy and set the timestamp.
        # There are many rows where CC and / or IQ are nan or None. To handle these cases, we will interpolate over
        # the data after inserting all the empty rows.
        empty_row = copy(df.iloc[0])
        empty_row[iq_band_col] = None
        empty_row[cc_band_col] = None
        empty_row[wind_speed_col] = None
        empty_row[wind_dir_col] = None

        # Create a "bad" row with the worst conditions to fill in missing gaps at beginning / ends of nights or
        # rows for entirely missing days.
        bad_row = copy(empty_row)
        bad_row[iq_band_col] = 1.0
        bad_row[cc_band_col] = 1.0
        bad_row[wind_speed_col] = 0.0
        bad_row[wind_dir_col] = 0.0

        # Store the data by night.
        data_by_night = {}

        for night_idx, pd_time_info in enumerate(zip(pd_time_starts, pd_time_ends)):
            pd_time_start, pd_time_end = pd_time_info
            time_slots_expected = c_time_slots_per_night[night_idx]

            # Night date
            pd_date_curr = pd_time_start.date()

            print(f'\nProcessing {site.name}, date {pd_date_curr}. Time slots expected: {time_slots_expected}:')

            # Counter for the current time to fill in missing entries.
            pd_time_curr = pd_time_start

            # The rows for the current night.
            night_rows = []

            # Get all the rows that fall between the twilights and sort the data by timestamp.
            night_df = df[
                df[time_stamp_col].between(pd_time_start, pd_time_end, inclusive="both")
            ]
            night_df.sort_values(by=time_stamp_col)

            # SPECIAL CASE: NO INPUT FOR NIGHT.
            if night_df.empty:
                print(f'\tNo data. Filling in with worst conditions.')
                # Loop from pd_start_time to pd_end_time, inserting the worst conditions.
                # TODO: Check to see what we should use for wind speed and direction.
                while pd_time_curr < pd_time_end:
                    new_row = copy(bad_row)
                    new_row[time_stamp_col] = pd_time_curr
                    night_rows.append(new_row)
                    pd_time_curr += pd_minute

                # Check to ensure the night contains the expected number of time slots.
                data_by_night[pd_date_curr] = night_rows
                time_slots_actual = len(data_by_night[pd_date_curr])
                if time_slots_expected != time_slots_actual:
                    print(f'\tActual time slots does not meet expected time slots: {time_slots_actual}')

                # Loop to next night.
                continue

            # *** REGULAR CASE: There is (some) data for date. ***
            prev_row = None

            # Fill in any missing data at beginning of night.
            start_gaps = 0
            pd_time_first_in_frame = night_df.iloc[0][time_stamp_col]
            if pd_time_curr < pd_time_first_in_frame:
                while pd_time_curr < pd_time_first_in_frame:
                    new_row = copy(bad_row)
                    new_row[time_stamp_col] = pd_time_curr
                    night_rows.append(new_row)
                    start_gaps += 1
                    pd_time_curr += pd_minute
                    prev_row = new_row
                print(f'\tData missing from start of night. Added {start_gaps} rows.')

            # Iterate over the rows.
            for idx, curr_row in night_df.iterrows():
                curr_row[cc_band_col] = val_to_float(curr_row[cc_band_col])
                curr_row[iq_band_col] = val_to_float(curr_row[iq_band_col])
                if pd.isna(curr_row[wind_dir_col]) or pd.isna(curr_row[wind_speed_col]):
                    print(f'\tMissing wind data for {site.name} {curr_row[time_stamp_col]}: '
                          f'speed={curr_row[wind_speed_col]}, dir={curr_row[wind_dir_col]}')

                # Check for jumps. These are in the data sets and not caused by the scripts.
                if prev_row is not None:
                    # Check for jumps of more than one bin in ImageQuality.
                    iq_prev = prev_row[iq_band_col]
                    iq_curr = curr_row[iq_band_col]
                    if (iq_curr is not None and iq_prev is not None
                            and iq_curr != iq_prev
                            and {iq_curr, iq_prev} not in [{1.0, 0.85}, {0.85, 0.7}, {0.7, 0.2}]):
                        t1 = prev_row[time_stamp_col].strftime('%H:%M')
                        t2 = curr_row[time_stamp_col].strftime('%H:%M')
                        print(f'\t* IQ jump from {t1} to {t2}: {iq_prev} to {iq_curr}')

                    # Check for jumps of more than one bin in CloudCover.
                    cc_prev = prev_row[cc_band_col]
                    cc_curr = curr_row[cc_band_col]
                    if (cc_curr is not None and cc_prev is not None
                            and cc_curr != cc_prev
                            and {cc_curr, cc_prev} not in [{1.0, 0.8}, {0.8, 0.7}, {0.7, 0.5}]):
                        t1 = prev_row[time_stamp_col].strftime('%H:%M')
                        t2 = curr_row[time_stamp_col].strftime('%H:%M')
                        print(f'\t* CC jump for {t1} to {t2}: {cc_prev} to {cc_curr}')

                # Get the timestamp for the current row and determine if there is missing data.
                pd_next_time = curr_row[time_stamp_col]
                gaps = int((pd_next_time - pd_time_curr).total_seconds() / 60)

                # NOTE: Linear interpolation on wind has been moved below.
                # Fill in any gaps in wind data between the prev_row and the curr_row with linear interpolation.
                # We will take another pass through the data to fill in CC and IQ entries due to the abundance
                # of nan and None entries.
                if gaps > 0:
                    # wind_dir = lerp_degrees(prev_row[wind_dir_col], curr_row[wind_dir_col], gaps)
                    # wind_speed = lerp(prev_row[wind_speed_col], curr_row[wind_speed_col], gaps)
                    ii = 0
                    while ii < gaps:
                        new_row = copy(empty_row)
                        new_row[time_stamp_col] = pd_time_curr
                        # new_row[wind_dir_col] = wind_dir[ii]
                        # new_row[wind_speed_col] = wind_speed[ii]
                        new_row[wind_dir_col] = None
                        new_row[wind_speed_col] = None
                        night_rows.append(new_row)
                        ii += 1
                        pd_time_curr += pd_minute

                # Add the current row and designate it to the prev_row.
                # print(curr_row)
                night_rows.append(curr_row)
                prev_row = curr_row
                pd_time_curr = curr_row[time_stamp_col] + pd_minute

            end_gaps = 0
            if pd_time_curr < pd_time_end:
                while pd_time_curr < pd_time_end:
                    new_row = copy(bad_row)
                    new_row[time_stamp_col] = pd_time_curr
                    night_rows.append(new_row)
                    end_gaps += 1
                    pd_time_curr += pd_minute
                print(f'\tData missing from end of night. Added {end_gaps} rows.')

            clean_gaps(night_rows,iq_band_col, "IQ", 1.0, enum_gaps(ImageQuality))
            clean_gaps(night_rows, cc_band_col, "CC", 1.0, enum_gaps(CloudCover))
            clean_gaps(night_rows, wind_speed_col, "WdSpd", 0.0, continuous_gaps(lerp))
            clean_gaps(night_rows, wind_dir_col, "WdDir", 0.0, continuous_gaps(lerp_degrees))

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

            # Check for off-by-one errors, which should only happen when we add terminating rows.
            time_slot_difference = abs(c_time_slots_per_night[night_idx] - len(night_rows))
            if time_slot_difference:
                # Get the expected values and convert to datetime to standardize.
                expected_first = c_time_starts[night_idx]
                expected_first.format = 'datetime'
                expected_last = c_time_ends[night_idx]
                expected_last.format = 'datetime'

                pd_actual_first = night_rows[0][time_stamp_col].to_pydatetime().replace(tzinfo=None)
                pd_actual_last = night_rows[-1][time_stamp_col].to_pydatetime().replace(tzinfo=None)
                actual_first = Time(pd_actual_first, scale='utc')
                actual_last = Time(pd_actual_last, scale='utc')

                print(f'\tExpected: {time_slots_expected}, Actual: {len(night_rows)}, '
                      f'difference={time_slot_difference}\n'
                      f'\tExpected first: {expected_first}, expected last: {expected_last}\n'
                      f'\tActual first:   {actual_first},   actual last:   {actual_last}')

                if abs(time_slot_difference) != 1:
                    raise ValueError(f'Difference in time slots for date {pd_date_curr} is {time_slot_difference}.')

                # We have one extra time slot in the data. Drop it.
                night_rows.pop()
                time_slot_difference = abs(c_time_slots_per_night[night_idx] - len(night_rows))
                print(f'\tPopped entry. Difference now: {time_slot_difference}.')

            # Copy the night rows to the array.
            data_by_night[pd_date_curr] = night_rows

        # Flatten the data back down to a table.
        flattened_data = []
        for rows in data_by_night.values():
            flattened_data.extend(rows)

        # Convert back to a pandas dataframe and store.
        modified_df = pd.DataFrame(flattened_data)
        modified_df.to_pickle(output_file_name, compression="bz2")

        print(f"*** Done processing site: {site.name} ***")


def main():
    sites = (
        Site.GN,
        Site.GS,
    )
    in_files = [
        os.path.join("data", f)
        for f in ("gn_wfs_filled_final.pickle.bz2", "gs_wfs_filled_final.pickle.bz2",)
    ]
    out_files = [
        os.path.join("data", f)
        for f in ("gn_weather_data.pickle.bz2", "gs_weather_data.pickle.bz2",)
    ]

    for site, in_file, out_file in zip(sites, in_files, out_files):
        process_files(site, in_file, out_file)


if __name__ == "__main__":
    main()
