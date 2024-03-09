# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import datetime

import pandas as pd
from lucupy.minimodel import Site

from pathlib import Path

__all__ = [
    'ROOT_DIR',
    'utc_time_stamp_col',
    'local_time_stamp_col',
    'night_time_stamp_col',
    'cc_band_col',
    'iq_band_col',
    'wv_band_col',
    'bg_band_col',
    'wind_speed_col',
    'wind_dir_col',
    'first_date',
    'last_date',
    'day',
]

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

utc_time_stamp_col = 'Time_Stamp_UTC'
local_time_stamp_col = 'Local_Time'
night_time_stamp_col = 'Local_Night'
cc_band_col = 'raw_cc'
iq_band_col = 'raw_iq'
wv_band_col = 'raw_wv'
bg_band_col = 'raw_bg'
wind_speed_col = 'WindSpeed'
wind_dir_col = 'WindDir'

# The dates of relevance.
# Might be able to use lucupy Semester / SemesterHalf for this, but this is easier.
# We want three semesters:
# Night of 2018-02-01 to 2018-07-31.
# Night of 2018-08-01 to 2019-01-31.
# Night of 2019-02-01 to 2019-07-31.
# We take local noon to be the breakpoints since it will definitely be in the day.
first_date = {
    Site.GN: datetime(2018, 2, 1, 12, 0, tzinfo=Site.GN.timezone),
    Site.GS: datetime(2018, 2, 1, 12, 0, tzinfo=Site.GS.timezone)
}
last_date = {
    Site.GN: datetime(2019, 8, 1, 11, 59, tzinfo=Site.GN.timezone),
    Site.GS: datetime(2019, 8, 1, 11, 59, tzinfo=Site.GS.timezone)
}

day = pd.Timedelta(days=1)
