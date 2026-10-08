# Public app dataset

`da_prices.csv.gz` contains the cleaned hourly DE-LU day-ahead price series
used by the interactive app. It covers 1,632 complete 24-hour delivery days
from 2022-01-01 through 2026-06-29. DST transition days are excluded because
the research LP uses fixed 24-hour windows.

Source of publication: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
The data are redistributed under CC BY 4.0. This repository transforms the
source series by deduplicating timestamps, hourly aggregation, date clipping,
and removal of incomplete delivery days. ENTSO-E does not sponsor or endorse
this project.
