from train_ml import train_ml
from train_mlp import train_mlp
from compare_N_explain import run_explanation_pipeline
from predict_new import predict_new_input


def main():
    print("Starting end-to-end pipeline...\n")

    # -------------------------------
    # Train ML Model
    # -------------------------------
    print("Training ML model (Random Forest)...")
    ml_result = train_ml()
    ml_metrics = ml_result["metrics"]
    rf_model = ml_result["model"]
    scaler = ml_result["scaler"]
    feature_columns = ml_result["feature_columns"]

    # -------------------------------
    # Train DL Model
    # -------------------------------
    print("Training DL model (MLP Classifier)...")
    dl_result = train_mlp()
    dl_metrics = dl_result["metrics"]
    mlp_model = dl_result["model"]

    # -------------------------------
    # Print Model Performance
    # -------------------------------
    print("\n========== MODEL PERFORMANCE ==========")

    print("\n--- Random Forest ---")
    print(f"Accuracy : {ml_metrics['accuracy']:.4f}")
    print(f"Precision: {ml_metrics['precision']:.4f}")
    print(f"Recall   : {ml_metrics['recall']:.4f}")
    print(f"F1 Score : {ml_metrics['f1']:.4f}")

    print("\n--- MLP Classifier ---")
    print(f"Accuracy : {dl_metrics['accuracy']:.4f}")
    print(f"Precision: {dl_metrics['precision']:.4f}")
    print(f"Recall   : {dl_metrics['recall']:.4f}")
    print(f"F1 Score : {dl_metrics['f1']:.4f}")

    print("\n========================================\n")

    # -------------------------------
    # Model Selection using F1
    # -------------------------------
    print("Comparing models using F1-score...")

    if ml_metrics["f1"] >= dl_metrics["f1"]:
        selected_model = rf_model
        selected_type = "ML"
        print("Selected Model: Random Forest")
    else:
        selected_model = mlp_model
        selected_type = "DL"
        print("Selected Model: MLP Classifier")

    # -------------------------------
    # NEW USER INPUT (Demo Sample)
    # -------------------------------
    print("\nRunning prediction on new applicant input...\n")

    prediction, confidence_value = predict_new_input(
        person_age=35,
        person_gender="male",
        person_education="Bachelor",
        person_income=60000,
        person_emp_exp=5,
        person_home_ownership="RENT",
        loan_amnt=15000,
        loan_intent="PERSONAL",
        loan_int_rate=12.5,
        loan_percent_income=0.25,
        cb_person_cred_hist_length=6,
        credit_score=720,
        previous_loan_defaults_on_file="No",
        model=selected_model,
        scaler=scaler,
        feature_columns=feature_columns,
        model_type=selected_type
    )

    # -------------------------------
    # Convert Prediction + Confidence
    # -------------------------------
    final_prediction_label = "Loan Approved" if prediction == 1 else "Loan Rejected"
    confidence = f"{confidence_value * 100:.2f}%"

    print(f"Final Prediction: {final_prediction_label}")
    print(f"Confidence: {confidence}")

    # -------------------------------
    # LLM Explanation
    # -------------------------------
    print("\nRunning LLM explanation...")
    run_explanation_pipeline(
        ml_metrics,
        dl_metrics,
        final_prediction_label,
        confidence
    )

    print("\nPipeline completed successfully.")
    print("Output saved to outputs/final_output.json")


if __name__ == "__main__":
    main()
