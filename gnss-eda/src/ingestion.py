from src.config import PARQUET_DIR, RAW_DIR
from src.database import get_connection


def sql_string(value):
    """Safely quote a string as a SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def convert_csv_to_parquet():

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    con = get_connection()

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    for csv_file in csv_files:
        output_file = PARQUET_DIR / f"{csv_file.stem}.parquet"

        if output_file.exists():
            print(f"Skipping existing: {output_file.name}")
            continue

        print(f"\nConverting {csv_file.name}")

        source = sql_string(csv_file.resolve())
        destination = sql_string(output_file.resolve())

        query = f"""
            COPY (
                SELECT *
                FROM read_csv_auto(
                    {source},
                    header = true,
                    sample_size = -1
                )
            )
            TO {destination}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """

        con.execute(query)

        print(f"Saved: {output_file}")

    con.close()


if __name__ == "__main__":
    convert_csv_to_parquet()
