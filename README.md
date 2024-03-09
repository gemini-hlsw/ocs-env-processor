# ocs-env-processor

Project author: [Sebastian Raaphorst](https://www.github.com/sraaphorst), 2024-03-08.

This consists of a pair of Python scripts to take the environmental data per site from
the OCS and process it for use with the [Gemini Automated Scheduler](https://www.github.com/gemini-hlsw/scheduler).

* `filter_dates.py`:
  * Takes the original data files and drops irrelevant rows (data that is not temporally relevant) and columns.
  * Makes the UTC timestamp field timezone-aware and adds a local timestamp field that is also timezone-aware.
* `env_processor.py`
  * Uses the output from `filter_dates.py`.
  * Ensures values for cloud cover, image quality, water vapor, and sky background are legal.
  * Adds a night date field so that data from the same night can be easily queried.
  * Traverses each night date group and removes entries where the IQ and CC do not change, thus leaving only entries
    corresponding to a weather change event.

Note that the scripts must be run in the order listed above to create the intermediary files to result in the final
files for consumption with the Gemini Automated Scheduler.

### Input

The input is a dataset per site:

* Gemini North: `gn_wfs_filled_final.pickle.bz2`
* Gemini South: `gs_wfs_filled_final.pickle.bz2`

created from the historical night data of the telescopes.

### Output

The output is a dataset per site:

* Gemini North: `gn_data.pickle.bz2`
* Gemini South: `gs_data.pickle.bz2`

for the Scheduler in validation mode.

### Note

This is a replacement for the now-defunct [Scheduler OCS Env Service](https://github.com/gemini-hlsw/scheduler-ocs-env).
