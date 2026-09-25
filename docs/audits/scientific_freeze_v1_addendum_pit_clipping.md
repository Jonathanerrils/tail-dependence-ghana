# Scientific Freeze v1 Addendum: Panel A Cedi PIT Clipping Audit

**Date:** 21 September 2026  
**Status:** Closed. No full GOF rerun required.

## Issue identified

Panel A Cedi PIT values were clipped in `src/tailrisk/marginals.py` to the interval
`[1e-6, 1 - 1e-6]`. Three Cedi PIT observations were therefore tied exactly at
`1e-6`:

- 2022-09-19
- 2023-02-21
- 2023-12-29

The frozen Panel A PIT remained unchanged throughout the audit:

- Shape: 3007 x 5
- Period: 2015-01-06 to 2026-07-15
- SHA256: `be42455c12bf2f4a9c7ef33494930ad396f569192f571386d099046a388119d1`

Panel B had no exact PIT ties.

## Recovered ordering

Using the saved Panel A standardized residual file
`outputs/tables/_resid_real.csv`, the true ordering of the three clipped Cedi
observations was recovered because the fitted marginal CDF is monotone.

Recovered ranks:

1. 2023-12-29, residual -113.05676009, rank 1
2. 2023-02-21, residual -91.8235450571, rank 2
3. 2022-09-19, residual -85.0803675326, rank 3

Under clipping, all three had average rank 2.

Only two pseudo-observations changed after recovering ranks 1, 2, and 3.
The maximum absolute pseudo-observation change was
`0.000332446808511`.

## Stress inference consequence

Lower-tail membership was unchanged at all pre-specified thresholds:

- q = 0.025: 0 membership changes
- q = 0.050: 0 membership changes
- q = 0.100: 0 membership changes

Therefore the clipping artifact does not alter which observations enter the
finite-tail concentration sets used by the stress analysis.

## Copula GOF consequence

Observed GOF sensitivity was evaluated for all 32 Panel A commodity-Cedi cells
(4 pairs x 8 copula families), without bootstrap reruns.

Maximum absolute observed-statistic change across the 32 cells:

- `max |Delta Sn| = 0.000262`

For the previously borderline Cocoa-Cedi Frank cell:

- Current Sn: `0.025892`
- Recovered-order Sn: `0.025892`
- Absolute Sn change: `0`
- Maximum absolute parameter change: approximately `0.000004`

Thus the borderline Cocoa-Cedi Frank GOF result was not created by the
1e-6 PIT clipping artifact.

## Student-t diagnostic

For Cocoa-Cedi, the fitted Student-t copula degrees of freedom changed from

- current: `nu = 11,617,958.115`
- recovered: `nu = 5,592,649.762`

while correlation changed only from

- `rho = 0.00645885`
to
- `rho = 0.00656525`

and the GOF statistic changed by only

- `|Delta Sn| = 8.81e-05`.

Both degrees-of-freedom estimates are extremely large. The Student-t copula is
therefore effectively at its Gaussian limit in this cell, and `nu` is weakly
identified numerically. The large absolute change in `nu` is not a material
change in fitted dependence.

## Frozen decision

1. Do not rerun the full 32-cell Panel A Cedi GOF bootstrap because of this issue.
2. Do not rerun stress inference because lower-tail membership is unchanged.
3. Retain the existing Panel A and Panel B headline classifications.
4. In the manuscript, describe the PIT clipping as a numerical safeguard and do
   not interpret the extremely large Cocoa-Cedi Student-t degrees-of-freedom
   estimate substantively.
5. If a targeted Cocoa-Cedi Frank high-resolution bootstrap is ever run, treat it
   as a Monte Carlo-precision check, not as a correction for PIT clipping.

**Audit status: CLOSED.**
