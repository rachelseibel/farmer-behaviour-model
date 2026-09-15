#!/usr/bin/env python3
"""Generate a synthetic cattle landscape for testing the model.

The real analysis runs on Cattle Tracing System holding records supplied by the
British Cattle Movement Service, and on farmer survey responses. Neither can be
redistributed (see DATA.md), so this script produces a synthetic landscape with
the same schema and broadly similar structure, which is enough to run the model
end to end and check that an installation works.

Nothing here is derived from the real data. Holding locations are drawn from a
clustered point process on a plain rectangle, herd sizes from a lognormal
distribution, and behavioural clusters from a spatially correlated field. The
numbers are plausible for Great Britain in order of magnitude only. Do not use
the output for inference.

Usage
-----
    python tools/make_sample_landscape.py --outdir sampledata --holdings 2000

Produces, in --outdir:

    grid_holdings.csv    nation, region, county, cell_id, holding_id,
                         easting, northing, cattle
    grid_cells.csv       cell_id, min_x, min_y, max_x, max_y, num_nodes
    behavioural_landscape.csv
                         nation, region, county, cattle, easting, northing,
                         iteration, cluster_id, sample_id
                         (one block of `holdings` rows per iteration)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# A plain rectangle in British National Grid metres. Not a real extent.
X0, X1 = 200_000.0, 500_000.0
Y0, Y1 = 200_000.0, 600_000.0

NATIONS = ["England", "Scotland", "Wales"]
REGIONS = ["North", "Midlands", "South", "East", "West"]


def clustered_points(rng: np.random.Generator, n: int, n_parents: int = 40,
                     spread: float = 12_000.0) -> tuple[np.ndarray, np.ndarray]:
    """Thomas-process-like clustering: parents scattered uniformly, offspring
    scattered normally around them. Cattle holdings are not uniform in space and
    a uniform landscape would flatter the model, so this keeps some clumping."""
    px = rng.uniform(X0, X1, n_parents)
    py = rng.uniform(Y0, Y1, n_parents)
    which = rng.integers(0, n_parents, n)
    x = np.clip(px[which] + rng.normal(0.0, spread, n), X0, X1)
    y = np.clip(py[which] + rng.normal(0.0, spread, n), Y0, Y1)
    return x, y


def correlated_field(rng: np.random.Generator, x: np.ndarray, y: np.ndarray,
                     length_scale: float = 40_000.0) -> np.ndarray:
    """A smooth spatial field, used to give behavioural clusters some spatial
    structure rather than assigning them independently. Built by smoothing white
    noise on a coarse grid and sampling it at the holding locations, which is
    cheap and good enough for a test fixture."""
    g = 64
    noise = rng.normal(size=(g, g))
    gx = np.linspace(X0, X1, g)
    gy = np.linspace(Y0, Y1, g)
    cell = max((X1 - X0) / g, (Y1 - Y0) / g)
    radius = max(1, int(length_scale / cell))
    kernel = np.exp(-0.5 * (np.arange(-3 * radius, 3 * radius + 1) / radius) ** 2)
    kernel /= kernel.sum()
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, noise)
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, smooth)
    ix = np.clip(np.searchsorted(gx, x) - 1, 0, g - 1)
    iy = np.clip(np.searchsorted(gy, y) - 1, 0, g - 1)
    v = smooth[iy, ix]
    return (v - v.mean()) / (v.std() or 1.0)


def build(n_holdings: int, n_iterations: int, cell_size: float,
          seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x, y = clustered_points(rng, n_holdings)
    # Herd sizes are strongly right-skewed: many small holdings, a few large.
    cattle = np.clip(rng.lognormal(mean=4.1, sigma=1.0, size=n_holdings), 1, 5000)
    cattle = cattle.round().astype(int)

    nation = rng.choice(NATIONS, n_holdings, p=[0.7, 0.2, 0.1])
    region = rng.choice(REGIONS, n_holdings)
    county = np.array([f"County_{i:02d}" for i in rng.integers(0, 40, n_holdings)])

    col = np.floor((x - X0) / cell_size).astype(int)
    row = np.floor((y - Y0) / cell_size).astype(int)
    n_col = int(np.ceil((X1 - X0) / cell_size))
    raw_cell = row * n_col + col
    # Cell ids must be contiguous from zero, as the real pipeline guarantees.
    uniq = np.sort(np.unique(raw_cell))
    remap = {c: i for i, c in enumerate(uniq)}
    cell_id = np.array([remap[c] for c in raw_cell])

    holdings = pd.DataFrame({
        "nation": nation,
        "region": region,
        "county": county,
        "cell_id": cell_id,
        "holding_id": np.arange(n_holdings),
        "easting": x.round(1),
        "northing": y.round(1),
        "cattle": cattle,
    })

    counts = holdings["cell_id"].value_counts()
    cells = pd.DataFrame({"cell_id": np.arange(len(uniq))})
    c_col = uniq % n_col
    c_row = uniq // n_col
    cells["min_x"] = X0 + c_col * cell_size
    cells["min_y"] = Y0 + c_row * cell_size
    cells["max_x"] = cells["min_x"] + cell_size
    cells["max_y"] = cells["min_y"] + cell_size
    cells["num_nodes"] = cells["cell_id"].map(counts).fillna(0).astype(int)

    # Behavioural landscape: one block of rows per iteration. Four clusters,
    # matching the four behavioural clusters of the elicitation study, assigned
    # by quartile of a spatially correlated field so that they come out clumped
    # rather than independent.
    blocks = []
    for it in range(n_iterations):
        field = correlated_field(rng, x, y)
        cuts = np.quantile(field, [0.25, 0.5, 0.75])
        cluster = np.digitize(field, cuts) + 1  # 1..4
        blocks.append(pd.DataFrame({
            "nation": nation,
            "region": region,
            "county": county,
            "cattle": cattle,
            "easting": x.round(1),
            "northing": y.round(1),
            "iteration": it,
            "cluster_id": cluster,
            "sample_id": rng.integers(0, 475, n_holdings),
        }))
    landscape = pd.concat(blocks, ignore_index=True)
    return holdings, cells, landscape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="sampledata")
    ap.add_argument("--holdings", type=int, default=2000,
                    help="number of synthetic holdings (default 2000)")
    ap.add_argument("--iterations", type=int, default=5,
                    help="behavioural landscape realisations (default 5)")
    ap.add_argument("--cell-size", type=float, default=10_000.0,
                    help="grid cell side in metres (default 10000)")
    ap.add_argument("--seed", type=int, default=20260915,
                    help="RNG seed; the same seed always gives the same landscape")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    holdings, cells, landscape = build(a.holdings, a.iterations, a.cell_size, a.seed)

    holdings.to_csv(os.path.join(a.outdir, "grid_holdings.csv"), index=False)
    cells.to_csv(os.path.join(a.outdir, "grid_cells.csv"), index=False)
    landscape.to_csv(os.path.join(a.outdir, "behavioural_landscape.csv"), index=False)

    print(f"Wrote synthetic landscape to {a.outdir}/")
    print(f"  grid_holdings.csv          {len(holdings):>7,} holdings")
    print(f"  grid_cells.csv             {len(cells):>7,} occupied cells")
    print(f"  behavioural_landscape.csv  {len(landscape):>7,} rows "
          f"({a.iterations} iterations)")
    print(f"  total cattle               {holdings['cattle'].sum():>7,}")
    print(f"  median herd size           {int(holdings['cattle'].median()):>7,}")
    print("\nSynthetic data. Not derived from CTS or survey records. "
          "Do not use for inference.")


if __name__ == "__main__":
    main()
