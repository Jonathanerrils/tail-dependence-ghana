"""Targeted diagnostic for the COVID-only Brent-WTI upper-tail result."""

import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np

from tailrisk import copulas


PIT = "outputs/tables/_pit_real.csv"

COVID = ("2020-03-01", "2020-06-30")
C24 = ("2024-01-01", "2024-12-31")

pit = pd.read_csv(PIT, index_col=0, parse_dates=True)
idx = pit.index

covid_flag = (idx >= COVID[0]) & (idx <= COVID[1])
c24_flag = (idx >= C24[0]) & (idx <= C24[1])

calm = pit[~(covid_flag | c24_flag)]
covid = pit[covid_flag]


def upper_details(frame, q):
    u = copulas.pseudo_obs(frame[["brent"]])[:, 0]
    v = copulas.pseudo_obs(frame[["wti"]])[:, 0]

    joint = (1 - u <= q) & (1 - v <= q)

    lam = joint.mean() / q

    detail = pd.DataFrame(
        {
            "brent_pit": frame["brent"].to_numpy(),
            "wti_pit": frame["wti"].to_numpy(),
            "brent_rank": u,
            "wti_rank": v,
            "joint_upper": joint,
        },
        index=frame.index,
    )

    return lam, int(joint.sum()), detail


rows = []

for q in (0.025, 0.05, 0.10):
    lc, nc, _ = upper_details(calm, q)
    ls, ns, detail = upper_details(covid, q)

    rows.append({
        "q": q, "n_calm": len(calm), "joint_calm": nc, "lambda_calm": lc,
        "n_covid": len(covid), "joint_covid": ns, "lambda_covid": ls,
        "difference": ls - lc,
    })

    print(f"\n=== q={q:.3f} ===")
    print(f"Calm:  joint={nc}, lambda={lc:.6f}")
    print(f"COVID: joint={ns}, lambda={ls:.6f}")
    print(f"Difference: {ls-lc:.6f}")

    print("\nCOVID joint upper-tail dates:")
    print(detail.loc[detail["joint_upper"],
                     ["brent_pit", "wti_pit", "brent_rank", "wti_rank"]].to_string())

summary = pd.DataFrame(rows)
print("\n=== SUMMARY ===")
print(summary.to_string(index=False))
summary.to_csv("outputs/tables/diagnostic_brent_wti_covid_upper.csv",
               index=False, float_format="%.6f", lineterminator="\n")

_, _, covid_detail = upper_details(covid, 0.05)
dates = covid_detail.loc["2020-04-15":"2020-04-24",
                         ["brent_pit", "wti_pit", "brent_rank", "wti_rank", "joint_upper"]]
print("\n=== 15-24 April 2020 ===")
print(dates.to_string())


def diff_after_dropping(drop_dates):
    stress = covid.drop(pd.to_datetime(drop_dates), errors="ignore")
    lc, _, _ = upper_details(calm, 0.05)
    ls, ns, _ = upper_details(stress, 0.05)
    return len(stress), ns, ls, ls - lc


tests = {
    "none": [],
    "drop_2020_04_20": ["2020-04-20"],
    "drop_2020_04_20_to_22": ["2020-04-20", "2020-04-21", "2020-04-22"],
}

print("\n=== DATE-REMOVAL SENSITIVITY, q=.05 upper ===")
for name, dates_ in tests.items():
    n, joint, lam, diff = diff_after_dropping(dates_)
    print(f"{name:25s} n={n:2d} joint={joint} lambda={lam:.6f} difference={diff:.6f}")

stress_wide = covid.loc[~((covid.index >= "2020-04-15") & (covid.index <= "2020-04-23"))]
lc, _, _ = upper_details(calm, 0.05)
ls, ns, _ = upper_details(stress_wide, 0.05)
print(f"{'drop_2020_04_15_to_23':25s} n={len(stress_wide):2d} joint={ns} "
      f"lambda={ls:.6f} difference={ls-lc:.6f}")