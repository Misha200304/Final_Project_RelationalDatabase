import argparse
import csv
import sqlite3
from pathlib import Path
from typing import List, Sequence


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def infer_sqlite_types(csv_path: Path, sample_size: int = 5000) -> List[str]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

        is_int = [True] * len(header)
        is_float = [True] * len(header)
        seen_non_empty = [False] * len(header)

        for i, row in enumerate(reader):
            if sample_size and i >= sample_size:
                break

            for idx, value in enumerate(row):
                value = value.strip()
                if value == "":
                    continue

                seen_non_empty[idx] = True

                if is_int[idx]:
                    try:
                        int(value)
                    except ValueError:
                        is_int[idx] = False

                if is_float[idx]:
                    try:
                        float(value)
                    except ValueError:
                        is_float[idx] = False

        types = []
        for idx in range(len(header)):
            if not seen_non_empty[idx]:
                types.append("TEXT")
            elif is_int[idx]:
                types.append("INTEGER")
            elif is_float[idx]:
                types.append("REAL")
            else:
                types.append("TEXT")

        return types


def convert_value(value: str, sqlite_type: str):
    value = value.strip()
    if value == "":
        return None

    if sqlite_type == "INTEGER":
        try:
            return int(value)
        except ValueError:
            return None

    if sqlite_type == "REAL":
        try:
            return float(value)
        except ValueError:
            return None

    return value


def create_schema(conn: sqlite3.Connection, table_name: str, header: Sequence[str], types: Sequence[str]) -> None:
    quoted_cols = [f"{quote_identifier(col)} {col_type}" for col, col_type in zip(header, types)]
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        + ", ".join(quoted_cols)
        + ")"
    )
    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
    conn.execute(ddl)


def create_indexes(conn: sqlite3.Connection, table_name: str, header: Sequence[str]) -> None:
    index_candidates = [
        "tourney_name",
        "surface",
        "winner_name",
        "loser_name",
        "year",
        "winner_id",
        "loser_id",
    ]

    for col in index_candidates:
        if col in header:
            idx_name = f"idx_{table_name}_{col}"
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {quote_identifier(idx_name)} "
                f"ON {quote_identifier(table_name)} ({quote_identifier(col)})"
            )


def import_csv_to_sqlite(csv_path: Path, db_path: Path, table_name: str, infer_sample: int) -> int:
    types = infer_sqlite_types(csv_path, sample_size=infer_sample)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f, sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")

        reader = csv.reader(f)
        header = next(reader)

        create_schema(conn, table_name, header, types)

        placeholders = ", ".join(["?"] * len(header))
        columns = ", ".join(quote_identifier(col) for col in header)
        insert_sql = (
            f"INSERT INTO {quote_identifier(table_name)} ({columns}) "
            f"VALUES ({placeholders})"
        )

        batch_size = 1000
        batch = []
        inserted = 0

        for row in reader:
            converted = [convert_value(value, col_type) for value, col_type in zip(row, types)]
            batch.append(converted)

            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                inserted += len(batch)
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
            inserted += len(batch)

        create_indexes(conn, table_name, header)
        conn.commit()
        return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a SQLite database from a tennis matches CSV file.")
    parser.add_argument("--csv", default="correct_master.csv", help="Path to source CSV file.")
    parser.add_argument("--db", default="tennis.db", help="Path to output SQLite database.")
    parser.add_argument("--table", default="matches", help="Target table name.")
    parser.add_argument(
        "--infer-sample",
        type=int,
        default=5000,
        help="How many rows to inspect for type inference (0 = full scan).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    db_path = Path(args.db)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    inserted = import_csv_to_sqlite(csv_path, db_path, args.table, args.infer_sample)
    print(f"Database created: {db_path}")
    print(f"Table: {args.table}")
    print(f"Rows imported: {inserted}")


if __name__ == "__main__":
    main()