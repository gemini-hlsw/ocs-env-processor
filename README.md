# ocs-env-processor

This is a Python script to take the environmental data per site from
the OCS, parse out the relevant dates, and then ensure that
an entry for every time slot for every date contains weather
information.

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
