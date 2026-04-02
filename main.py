from src.data_prep import prepare_clean_data, save_clean_data
from src.features import build_modeling_table, save_modeling_table
from src.train import train_pipeline


def main() -> None:
    print("Step 1: Preparing cleaned transaction data...")
    clean_df = prepare_clean_data()
    save_clean_data(clean_df)
    print(f"Cleaned data shape: {clean_df.shape}")

    print("Step 2: Building customer modeling table...")
    modeling_df = build_modeling_table(clean_df)
    save_modeling_table(modeling_df)
    print(f"Modeling table shape: {modeling_df.shape}")

    print("Step 3: Training two-stage CLV system...")
    metrics = train_pipeline()
    print("Training complete.")
    print("Final metrics:")
    print(metrics["final_two_stage_metrics"])


if __name__ == "__main__":
    main()
