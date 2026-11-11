# Render 02_table_<Type>.md for all 12 record types from defs + profiles.
# Also rebuilds column_coverage.csv (one row per table.column).
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defs_raw import RAW_DEFS
from defs_tables import (
    STATUS_DEFS,
    FIX_DEFS,
    AGC_DEFS,
    NAV_DEFS,
    ORIENTATION_DEFS,
    GRV_DEFS,
    FOP_DEFS,
    ACCEL_DEFS,
    UNCAL_ACCEL_DEFS,
    UNCAL_GYRO_DEFS,
    UNCAL_MAG_DEFS,
)

OUT = "/home/tempest/Documents/ISRO_Project/docs/EDA"
PROF = os.path.join(OUT, "scratch", "profiles")

TABLES = {
    "Raw": (
        RAW_DEFS,
        "per-signal GNSS measurement record (one row per tracked signal per ~1 Hz epoch)",
    ),
    "Status": (
        STATUS_DEFS,
        "per-signal satellite status record (milestone-based, ~1 Hz batches)",
    ),
    "Fix": (
        FIX_DEFS,
        "position fix record (Android Location objects from FLP/GPS/NLP)",
    ),
    "Agc": (
        AGC_DEFS,
        "clock + automatic gain control block (1 Hz, per constellation/band)",
    ),
    "Nav": (NAV_DEFS, "navigation-message chunk record (variable-arity byte arrays)"),
    "OrientationDeg": (
        ORIENTATION_DEFS,
        "Euler-angle orientation record (yaw/roll/pitch, ~125 Hz)",
    ),
    "GameRotationVector": (
        GRV_DEFS,
        "TYPE_GAME_ROTATION_VECTOR quaternion (accel+gyro fusion, ~125 Hz)",
    ),
    "FOP": (
        FOP_DEFS,
        "fused orientation provider quaternion + magnetic heading (~125 Hz)",
    ),
    "Accel": (ACCEL_DEFS, "calibrated accelerometer record (~200 Hz, 29 sessions)"),
    "UncalAccel": (UNCAL_ACCEL_DEFS, "uncalibrated accelerometer record (60 sessions)"),
    "UncalGyro": (UNCAL_GYRO_DEFS, "uncalibrated gyroscope record (~200 Hz)"),
    "UncalMag": (UNCAL_MAG_DEFS, "uncalibrated magnetometer record (~55 Hz)"),
}


def fnum(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, int) or (
        isinstance(x, float) and abs(x) >= 1e6 or x == int(x) and abs(x) < 1e15
    ):
        return f"{x:,.0f}" if abs(x) >= 1e3 or x == int(x) else f"{x:g}"
    return f"{x:.{nd}g}"


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "—"


def col_block(c, nrows):
    L = []
    n_null = c.get("n_null", 0)
    nn = c.get("n", nrows) or 1
    L.append(
        f"- **DuckDB type:** `{c.get('dtype', '?')}`; **nulls:** {n_null:,} ({pct(n_null, nn)})"
    )
    nd_ = c.get("n_distinct")
    if nd_ is not None:
        L.append(
            f"- **Distinct:** {nd_:,}"
            + ("  *(exact)*" if nd_ <= 100 else "  *(approx_count_distinct)*")
        )
    for key, lab in (
        ("mn", "min"),
        ("mx", "max"),
        ("mean", "mean"),
        ("std", "std"),
        ("p50", "p50"),
        ("p95", "p95"),
        ("p99", "p99"),
    ):
        if c.get(key) is not None:
            L.append(f"- **{lab}:** {fnum(c[key])}")
    for key, lab in (
        ("n_zero", "zeros"),
        ("n_neg", "negatives"),
        ("n_nan", "NaNs"),
        ("n_inf", "infs"),
    ):
        if c.get(key):
            L.append(f"- **{lab}:** {c[key]:,} ({pct(c[key], nn)})")
    top = c.get("top10")
    if top:
        cells = ", ".join(f"`{v}` = {n:,}" for v, n in top[:8])
        L.append(f"- **Top values:** {cells}")
    return L


def main():
    cov_rows = ["table,column,dtype,documented_status,notes"]
    for table, (defs, desc) in TABLES.items():
        pj = json.load(open(os.path.join(PROF, f"{table}.json")))
        cols = pj["columns"]
        # sanity: defs cover exactly the profiled columns
        missing = set(cols) - set(defs)
        extra = set(defs) - set(cols)
        if missing or extra:
            print(
                f"[WARN {table}] defs/profile mismatch: missing={missing} extra={extra}"
            )
        nrows = pj["n_rows"]
        t = pj["time"]
        L = [
            f"# 02 — Record type `{table}`",
            "",
            f"**What it is:** {desc}.",
            "",
            f"**Volume [OBSERVED]:** {nrows:,} rows across **{pj['n_sessions']} sessions** (files containing this type); "
            + (
                f"time range {t['tmin']:,} → {t['tmax']:,} (UTC ms)."
                if t
                else "no wall-clock column is declared for this record type (Nav carries no timestamp of its own — its time context comes from the surrounding Raw/Status rows)."
            ),
            "Every declared column is documented below in the logger's header order — no exceptions. "
            "Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/%s.json` "
            "(approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`)."
            % table,
            "",
            "---",
            "",
        ]
        for name in cols:  # header order preserved by the profiler
            d = defs.get(
                name,
                dict(
                    what="(definition missing)",
                    why="",
                    use="",
                    caveats="",
                    status="TODO",
                ),
            )
            c = cols[name]
            L.append(f"## `{name}`")
            L.append("")
            L.append(f"**What:** {d['what']}")
            L.append("")
            L.append(f"**Why it matters:** {d['why']}")
            L.append("")
            L.append(f"**Use / project relevance:** {d['use']}")
            L.append("")
            L.append(f"**Caveats:** {d['caveats']}")
            L.append("")
            L.append(f"**Status:** {d['status']}")
            L.append("")
            L.append("**Observed in this corpus:**")
            L.extend(col_block(c, nrows))
            if d.get("observed_frequencies"):
                L.append(f"- **Bitmask decode:** {d['observed_frequencies']}")
            L.append("")
            cov_rows.append(
                f"{table},{name},{c.get('dtype', '?')},documented,see 02_table_{table}.md"
            )
        with open(os.path.join(OUT, f"02_table_{table}.md"), "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"[written] 02_table_{table}.md  ({len(cols)} columns)")
    with open(os.path.join(OUT, "column_coverage.csv"), "w") as f:
        f.write("\n".join(cov_rows) + "\n")
    print(f"column_coverage.csv: {len(cov_rows) - 1} table.column rows")


if __name__ == "__main__":
    main()
