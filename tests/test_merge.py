"""Test GNSS merge pipeline."""

import tempfile
from pathlib import Path
import duckdb
import pytest


def test_csv_column_mapping():
    """Verify DuckDB CSV columns are 0-indexed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("col1,col2,col3\n1,2,3\n")
        f.flush()

        con = duckdb.connect()
        # header=true uses actual column names
        res = con.execute(f"SELECT * FROM read_csv('{f.name}', header=true)").df()
        assert "col1" in res.columns

        # header=false generates column0, column1, etc
        res = con.execute(
            f"SELECT * FROM read_csv('{f.name}', header=false, skip=1)"
        ).df()
        assert "column0" in res.columns
        assert "column1" in res.columns

        Path(f.name).unlink()


def test_asof_join_session_isolation():
    """Verify ASOF JOIN respects session boundaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data: 2 sessions with overlapping timestamps
        con = duckdb.connect()

        con.execute("""
            CREATE TABLE status (ts BIGINT, sess VARCHAR, val INT);
            INSERT INTO status VALUES 
                (1000, 's1', 10), (2000, 's1', 20),
                (1500, 's2', 30), (2500, 's2', 40);
        """)

        con.execute("""
            CREATE TABLE orient (ts BIGINT, sess VARCHAR, angle INT);
            INSERT INTO orient VALUES 
                (900, 's1', 100), (1900, 's1', 200),
                (1400, 's2', 300), (2400, 's2', 400);
        """)

        # Without session isolation - wrong matches
        bad = con.execute("""
            SELECT s.ts, s.sess, o.sess as o_sess, o.angle
            FROM status s
            ASOF JOIN orient o ON s.ts >= o.ts
        """).df()

        # s2 at 1500 would match s1's orientation at 900 (wrong!)
        cross_session = bad[(bad["sess"] != bad["o_sess"])].shape[0]
        assert cross_session > 0, "Test setup should allow cross-session matches"

        # With session isolation - correct matches
        good = con.execute("""
            SELECT s.ts, s.sess, o.angle
            FROM status s
            ASOF JOIN orient o ON s.sess = o.sess AND s.ts >= o.ts
        """).df()

        # All matches should be within same session
        assert good.shape[0] == 4
        assert good.loc[good["sess"] == "s1", "angle"].tolist() == [100, 200]
        assert good.loc[good["sess"] == "s2", "angle"].tolist() == [300, 400]


def test_duplicate_handling():
    """Verify deterministic duplicate orientation handling."""
    con = duckdb.connect()

    con.execute("""
        CREATE TABLE orient (ts BIGINT, nano BIGINT, val INT);
        INSERT INTO orient VALUES 
            (1000, 100, 1), (1000, 200, 2), (1000, 300, 3);
    """)

    # Keep row with max nano (last received)
    dedup = con.execute("""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ts ORDER BY nano DESC) as rn
            FROM orient
        ) WHERE rn = 1
    """).df()

    assert dedup.shape[0] == 1
    assert dedup["val"].iloc[0] == 3  # highest nano wins


def test_output_schema():
    """Verify output has required columns for downstream models."""
    required = {
        "UnixTimeMillis",
        "Cn0DbHz",
        "AzimuthDegrees",
        "ElevationDegrees",
        "yawDeg",
        "pitchDeg",
        "rollDeg",
        "CalibrationAccuracy",
        "session_id",
        "device_id",
        "signal_id",
        "orientation_dt_ms",
        "ConstellationType",
        "Svid",
        "CarrierFrequencyHz",
    }

    con = duckdb.connect()
    actual = set(
        con.execute(
            "SELECT * FROM read_parquet('results/merged_data/combined_corpus.parquet') LIMIT 0"
        )
        .df()
        .columns
    )

    missing = required - actual
    assert not missing, f"Missing columns: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
