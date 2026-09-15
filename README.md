# Farmer behaviour model: livestock disease transmission with spatially heterogeneous vaccination behaviour

Model code for Chapter 4 of Rachel Seibel's PhD thesis (University of Warwick,
MathSys CDT): a spatial stochastic model of a fast-spreading cattle pathogen in
Great Britain, in which farmer vaccination behaviour is downscaled from survey
responses to individual holdings and its spatial arrangement is varied.

**No farm-level data is included here.** See [DATA.md](DATA.md) before doing
anything else.

## What is in the repository

    src/                model code
      landscape.py        holdings, grid cells, spatial structure
      behaviour.py        assign holdings to behavioural clusters and
                          vaccination triggers
      vaccination.py      vaccination uptake and timing
      transmission.py     kernel and cell-to-cell transmission probabilities
      seeding.py          outbreak seeding
      simulation.py       the epidemic simulation itself
      parameters.py       parameter dictionaries for a run
      run_model.py        single-run entry point
      run_model_cluster_version.py
                          HPC array-job entry point
      run_post_processing.py, run_move_files.py
                          summarise and collect outputs
      slurm/              example job configuration
    tools/
      make_sample_landscape.py
                          generate a synthetic landscape for testing

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy geopandas shapely

# Synthetic landscape, since the real inputs cannot be shared
python tools/make_sample_landscape.py --outdir sampledata --holdings 2000
```

Then run `src/run_model.py`. Paths default to the repository's own
`sampledata/` folder, so no editing is needed for a test run. To point the model
at real inputs, set the environment variables instead:

```bash
export LIVESTOCK_INPUT_DIR=/path/to/grid/inputs
export LIVESTOCK_LANDSCAPE_DIR=/path/to/behavioural/landscapes
export LIVESTOCK_COUNTY_NAMES=/path/to/holdings_counties.csv
export LIVESTOCK_STATUS_DIR=/path/to/status
```

## Reproducing the thesis results

You cannot, from this repository alone, because the inputs are restricted. With
the real data in place the analysis is:

1. Build the holding grid and behavioural landscape realisations under each of
   the five downscaling methods (Great Britain, nation, county, interpolation,
   blocking).
2. Run the epidemic model across seed counties and landscape realisations.
3. Post-process to cumulative outbreak and vaccination measures.

Parameter values are given in Chapter 4 of the thesis.

## Citing

Seibel, R. L. (2026). *Who adopts, and where? Modelling human disease control
behaviour across pathogen systems and host species.* PhD thesis, University of
Warwick.

The behavioural clusters build on:

- Hill, E. M. et al. (2023). Incorporating heterogeneity in farmer disease
  control behaviour into a livestock disease transmission model.
- Prosser, N. S. et al. (2022). Cattle farmer psychosocial profiles and their
  association with control strategies for bovine viral diarrhoea.

## Licence

MIT, see [LICENSE](LICENSE). The licence covers the code only. It does not grant
any rights over the data described in DATA.md.
