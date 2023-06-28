# ocs-env-processor

This consists of a pair of Python scripts to take the environmental data per site from
the OCS, parse out the relevant dates and columns, and then ensure that
an entry for every time slot for every date contains weather
information.

* `filter_dates.py` takes the original pandas dataframe files and filters out irrelevant dates and columns.
* `env_processor.py` takes the filtered pandas dataframe files and adds data missing for days or for gaps.

###  If a night contains no information

The night is populated with the default worst weather information possible.

### If a night is missing entries at the beginning or end

The missing entries are populated with the default worst weather information possible.

### If a night is missing intermediate entries

Linear interpolation is done between the existing entries to obtain
weather data for the missing entries.

### Output

The output is a new and complete dataset per site for the Scheduler in validation mode.

This is a replacement for the [Scheduler OCS Env Service](https://github.com/gemini-hlsw/scheduler-ocs-env).
