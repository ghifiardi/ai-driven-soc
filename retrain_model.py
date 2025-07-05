import argparse
import logging
import pandas as pd
import joblib
from google.cloud import firestore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_TRAINING = 4

def fetch_training_data(db: firestore.Client) -> list:
    """Fetches all documents from the 'ground_truth_alerts' collection."""
    logger.info("Fetching training data from 'ground_truth_alerts' collection...")
    docs = db.collection("ground_truth_alerts").stream()
    data = [doc.to_dict() for doc in docs]
    logger.info(f"Successfully fetched {len(data)} labeled documents.")
    return data

def retrain_model(project_id: str, model_output_path: str):
    """
    Fetches labeled data from Firestore, trains a new supervised model,
    and saves it to a file.
    """
    try:
        db = firestore.Client(project=project_id)
    except Exception as e:
        logger.critical(f"Failed to initialize Firestore client: {e}")
        return

    # 1. Fetch and prepare the data
    training_data = fetch_training_data(db)
    if len(training_data) < MIN_SAMPLES_FOR_TRAINING:
        logger.warning(f"Insufficient data for retraining. Need at least {MIN_SAMPLES_FOR_TRAINING} samples, but found {len(training_data)}. Aborting.")
        return

    df = pd.json_normalize(training_data, sep='_')

    # Filter out any records that might not be binary 'true_positive' or 'false_positive'
    df = df[df['classification'].isin(['true_positive', 'false_positive'])]

    # Define features (X) and target (y)
    # We use the original anomaly score and key categorical features from the raw alert.
    features = ['raw_alert_data_anomaly_score', 'raw_alert_data_method_name', 'raw_alert_data_service_name']
    target = 'classification'

    X = df[features]
    y = df[target].apply(lambda x: 1 if x == 'true_positive' else 0) # Encode target labels

    # --- Data Validation for Stratification ---
    # Ensure we have at least two samples in each class to perform a stratified split.
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        logger.critical("Model retraining aborted: The dataset is not suitable for a stratified split.")
        logger.critical(f"The least populated class has only {class_counts.min()} sample(s). A minimum of 2 is required.")
        logger.critical("Please label more alerts, ensuring there are at least 2 'true_positive' and 2 'false_positive' examples.")
        return

    logger.info(f"Data prepared for training. Feature columns: {features}")

    # 2. Define the preprocessing and model pipeline
    # We need to one-hot encode the categorical features.
    categorical_features = ['raw_alert_data_method_name', 'raw_alert_data_service_name']
    numeric_features = ['raw_alert_data_anomaly_score']

    # Create a preprocessor to handle different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the full pipeline with the preprocessor and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # 3. Split data and train the model
    test_size_ratio = 0.2
    n_samples = len(y)
    n_classes = len(y.unique())

    # --- Data Validation for Test Set Size ---
    # Ensure the test set is large enough to contain at least one sample from each class.
    if n_samples * test_size_ratio < n_classes:
        logger.critical("Model retraining aborted: The dataset is too small for the configured test split.")
        logger.critical(f"With {n_samples} samples and a test size of {test_size_ratio*100}%, the test set would have only {int(n_samples * test_size_ratio)} sample(s).")
        logger.critical(f"This is too small to represent all {n_classes} classes. A minimum of {n_classes} samples are required in the test set.")
        logger.critical("Please label more alerts to increase the dataset size.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42, stratify=y)

    logger.info(f"Training new RandomForestClassifier on {len(X_train)} samples...")
    model_pipeline.fit(X_train, y_train)
    logger.info("Model training complete.")

    # 4. Evaluate the new model on the test set
    logger.info("Evaluating newly trained model on the test set:")
    y_pred = model_pipeline.predict(X_test)
    print("\n--- New Model Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['false_positive', 'true_positive']))
    print("-------------------------------------\n")

    # 5. Save the entire pipeline (preprocessor + model)
    try:
        joblib.dump(model_pipeline, model_output_path)
        logger.info(f"New model pipeline successfully saved to: {model_output_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {model_output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the ADA detection model using ground truth data from Firestore.")
    parser.add_argument("--project_id", type=str, default="chronicle-dev-2be9", help="The Google Cloud project ID.")
    parser.add_argument("--output_path", type=str, default="supervised_model_v1.joblib", help="Path to save the trained model file.")
    args = parser.parse_args()

    retrain_model(project_id=args.project_id, model_output_path=args.output_path)
