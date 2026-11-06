import polars as pl
import polars.selectors as selectors
from src.config import RAW_TRAIN_DATA_DIR, TARGET_PATH

def save_conftest_data():
    data_lazy = (
        pl.scan_parquet(RAW_TRAIN_DATA_DIR)
        .with_columns(
            selectors.integer().exclude("id").cast(pl.Int16),
            pl.col("id").cast(pl.Int32),
            selectors.float().cast(pl.Float16),
        )
    )

    target_lazy = pl.scan_csv(
        TARGET_PATH,
        schema_overrides={"id": pl.Int32, "flag": pl.Int8},
    )

    data = (
        data_lazy
        .join(target_lazy, on="id", how="inner")
        .collect(engine="streaming")
        .sample(n=1_00, seed=1)
    )

    print(f"Data load with {len(data)} rows")
    data.write_parquet("conftest_data.parquet")
    print("data saved")


if __name__ == "__main__":
    save_conftest_data()