# Data availability

**No farm-level data is included in this repository, and none may be added to
it.**

## What the published analysis used, and why it is not here

The analysis reported in Chapter 4 of the thesis uses three inputs that cannot
be redistributed:

| Input | Source | Why it is not here |
|---|---|---|
| Cattle holding records: locations, herd sizes, county | Cattle Tracing System, maintained by the British Cattle Movement Service | Supplied under a data sharing agreement that does not permit redistribution. |
| Farmer elicitation responses (60 cattle farmers in Great Britain, April–June 2022) | Collected and reported by Hill et al. (2023) | Not ours to release; request from the original authors. |
| Bovine viral diarrhoea survey responses (475 UK cattle farmers, 13 July – 5 October 2020) | Collected and reported by Prosser et al. (2022) | As above. |

This extends to anything derived from them. Gridded holding counts, per-cell
cattle totals, pairwise distance lookups, transmission probability matrices and
behavioural landscape realisations are all derived products of the holding
records, and are excluded by `.gitignore` for that reason. If you are working
from a clone with real data in place, check `git status` before committing.

To obtain the real inputs, apply to the data holders directly. Access to CTS
extracts is arranged through the British Cattle Movement Service; the survey
data should be requested from the authors of the two studies above.

## What is here instead

`tools/make_sample_landscape.py` generates a synthetic landscape with the same
schema, so that the model can be installed, run end to end, and checked without
any restricted data:

```bash
python tools/make_sample_landscape.py --outdir sampledata --holdings 2000
```

The generator is seeded, so the same seed always produces the same landscape.

**The synthetic data is not derived from the real data in any way.** Holding
locations come from a clustered point process on a plain rectangle, herd sizes
from a lognormal distribution, and behavioural clusters from a smoothed random
field. The values are of a plausible order of magnitude for Great Britain and
nothing more. They reproduce the *shape* of the inputs, not their content.

Results produced from the synthetic landscape are not comparable with any result
in the thesis, and should not be used for inference about cattle disease in
Great Britain.

## Files the model expects

| File | Columns |
|---|---|
| `grid_holdings.csv` | `nation, region, county, cell_id, holding_id, easting, northing, cattle` |
| `grid_cells.csv` | `cell_id, min_x, min_y, max_x, max_y, num_nodes` |
| behavioural landscape | `nation, region, county, cattle, easting, northing, iteration, cluster_id, sample_id`, with one block of rows per iteration |

Eastings and northings are British National Grid metres. `cell_id` must be
contiguous from zero.
