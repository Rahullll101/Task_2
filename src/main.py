from train_ml import train_ml
from train_mlp import train_mlp
from compare_N_explain import run_explanation_pipeline

def main():
    print("Starting end-to-end pipeline...\n")

    print("Training ML model (Random Forest)...")
    ml_metrics = train_ml()
    print("ML Metrics:", ml_metrics, "\n")

    print("Training DL model (MLP / ANN)...")
    dl_metrics = train_mlp()
    print("DL Metrics:", dl_metrics, "\n")

    print("Running comparison + LLM explanation...")
    run_explanation_pipeline(ml_metrics, dl_metrics)

    print("\n Pipeline completed successfully")
    print("Output saved to outputs/final_output.json")

if __name__ == "__main__":
    main()
