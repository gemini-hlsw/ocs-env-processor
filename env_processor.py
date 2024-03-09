# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2

import pandas as pd

from definitions import (bg_band_col, cc_band_col, day, iq_band_col, local_time_stamp_col, night_time_stamp_col,
                         wv_band_col, ROOT_DIR)


def process_dataset(input_file_name: str, output_file_name: str) -> None:
    """
    Process a pandas dataset to make it suitable for use with the OcsEnv service in the Scheduler.
    The idea is that we want to:
    1. Fix any issues with CC, IQ, WV, or SB/BG.
    2. Add a night_time_stamp_col with the timestamp of the night (timezone-aware, local) that the
       weather change occurred.
    3. Eliminate any rows that do not represent a weather change, i.e. they occur in the same night and
       have the same CC and IQ as the previous row.
    """
    with bz2.BZ2File(input_file_name, 'rb') as input_file:
        df = pd.read_pickle(input_file)

        # Fix CC, IQ, WV, and SB entries.
        # Note that some entries have a 90 for CC, which is not a valid value. Raise to 100 in this case.
        df[cc_band_col] = df[cc_band_col].apply(lambda x: 1.0 if pd.isna(x) or x == 90.0 else x / 100)
        df[iq_band_col] = df[iq_band_col].apply(lambda x: 1.0 if pd.isna(x) else x / 100)
        df[wv_band_col] = df[wv_band_col].apply(lambda x: 1.0 if pd.isna(x) else x / 100)
        df[bg_band_col] = df[bg_band_col].apply(lambda x: 1.0 if pd.isna(x) else x / 100)

        def create_night_time_stamp(curr_row):
            """
            Create a night_time_stamp for the given row.
            This is a timezone-aware time stamp where a night is considered the period from:
               noon (inclusive)
            on one day to:
               noon (exclusive)
            on the following day. Thus, if a timestamp has a value of hour < 12, we subtract a day from it to
            associate it with the proper night.
            """
            time_stamp = curr_row[local_time_stamp_col]
            if time_stamp.hour < 12:
                time_stamp = time_stamp - day
            return time_stamp.replace(hour=12, minute=0, second=0, microsecond=0)

        # Add a column corresponding to a time stamp for the night.
        # This is timezone-aware and set to noon in the local timezone.
        df[night_time_stamp_col] = df.apply(create_night_time_stamp, axis=1)

        def filter_rows(group, night_timestamp):
            """
            Filter the group of rows so that in a block of consecutive weather information where CC and IQ
            are the same, we eliminate all but the first row.
            We also preserve the night_time_stamp_col that was used to group the rows.
            """
            last_row = None
            filtered_rows = []

            for _, curr_row in group.iterrows():
                if (last_row is None
                        or curr_row[cc_band_col] != last_row[cc_band_col]
                        or curr_row[iq_band_col] != last_row[iq_band_col]):
                    # We want to keep the night_time_stamp_col's value that was used for grouping, i.e. night_timestamp.
                    curr_row[night_time_stamp_col] = night_timestamp
                    filtered_rows.append(curr_row)
                    last_row = curr_row
            return pd.DataFrame(filtered_rows)

        # Sort, group, and apply the filter_rows function while passing the night timestamp.
        # We pass this so we can keep the value.
        filtered_df = (df.sort_values(local_time_stamp_col)
                       .groupby(night_time_stamp_col, group_keys=False)
                       .apply(lambda x: filter_rows(x, x.name), include_groups=False))

        # Debug: print the results.
        # for idx, row in filtered_df.iterrows():
        #     night_date = row[night_time_stamp_col].strftime('%Y-%m-%d')
        #     print(f'{idx}  {night_date}   {row[local_time_stamp_col]}   {row[cc_band_col]}   {row[iq_band_col]}')

        # Write the new dataframe as the final output to be used by the OcsEnv service of the Automated Scheduler.
        filtered_df.to_pickle(output_file_name, compression="bz2")


def main():
    gn_data = (
        ROOT_DIR / 'data' / 'gn_filtered.pickle.bz2',
        ROOT_DIR / 'data' / 'gn_data.pickle.bz2'
    )
    gs_data = (
        ROOT_DIR / 'data' / 'gs_filtered.pickle.bz2',
        ROOT_DIR / 'data' / 'gs_data.pickle.bz2'
    )

    for in_file, out_file in [gn_data, gs_data]:
        process_dataset(in_file, out_file)


if __name__ == "__main__":
    main()
