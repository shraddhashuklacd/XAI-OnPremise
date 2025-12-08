import os
import pandas as pd
from xai_runner.runner import ExplainableModelRunner
from xai_runner.config import DATA_DIR


def auto_select_dataset() -> str:
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in DATA_DIR: {DATA_DIR}. "
            "Please add at least one .csv file."
        )

    files.sort()
    selected = files[0]
    full_path = os.path.join(DATA_DIR, selected)

    print("Explainable Driver Engine")
    print(f"Dataset Found : {selected}")
    print(f"Path          : {full_path}\n")

    return full_path


def main() -> None:
    try:
        dataset_path = auto_select_dataset()
    except FileNotFoundError as e:
        print("\nERROR:", str(e))
        return

    df = pd.read_csv(dataset_path)
    print(f"Loaded Dataset | Shape = {df.shape}\n")

    runner = ExplainableModelRunner(df)

    runner.select_target_column()
    runner.detect_feature_types()
    runner.preprocess_data()
    runner.confirm_problem_type()
    runner.build_final_model_matrices()
    runner.run_bivariate_drivers()
    runner.fit_global_model_for_explainability()
    runner.run_shap_explainer(max_rows=None)
    runner.summarize()

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
