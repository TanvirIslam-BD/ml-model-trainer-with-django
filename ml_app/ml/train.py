import logging
import os
import re
import pandas as pd
from django.http import JsonResponse
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from django_ml_app import settings
from ml_app.ml.autotune import model_train_with_auto_tune
from ml_app.ml.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, DATE_FEATURES, MODEL_PATH, PRICE_FEATURES
from ml_app.ml.models import TrainedFeatures, TrainingHistory
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc

logger = logging.getLogger(__name__)

MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'ml_app', 'saved_models', 'hist_gradient_boosting_model.pkl')
TRAINING_DATA_PATH = os.path.join(settings.BASE_DIR, 'ml_app', 'data', 'training_data.csv')


def train_model(data):

    #extract labelColumn
    label_column = data['labelColumn'].strip()
    file_path = data['file_path']

    data, feature_matrix_x, target_vector_y = load_and_prepare_data(data)

    model, target_vector_y_test, feature_matrix_x_test, best_tune_params = initiate_model_training(feature_matrix_x, target_vector_y)

    performance_metrics = calculate_model_performance(feature_matrix_x_test, target_vector_y_test, model)

    save_training_data(performance_metrics, best_tune_params)

    return model, *performance_metrics.values()



def save_training_data(performance_metrics, best_tune_params):
    # Save the data to the TrainingHistory model
    training_history = TrainingHistory.objects.create(
        learning_rate=best_tune_params["learning_rate"],
        max_iter=best_tune_params["max_iter"],
        max_leaf_nodes=best_tune_params["max_leaf_nodes"],
        min_samples_leaf=best_tune_params["min_samples_leaf"],
        accuracy=performance_metrics["accuracy"],
        precision=performance_metrics["precision"],
        recall=performance_metrics["recall"],
        f1_score=performance_metrics["f1_score"],
        roc_auc=performance_metrics.get("roc_auc"),  # Check if roc_auc is present
        confusion_matrix=performance_metrics["confusion"],
        classification_report=performance_metrics["report"],
        feature_importance=performance_metrics["feature_importance_data"]
    )
    return training_history  # Return the saved object for further use if needed


def calculate_model_performance(feature_matrix_x_test, y_test, model):

    target_vector_y_prediction = model.predict(feature_matrix_x_test)

    accuracy = accuracy_score(y_test, target_vector_y_prediction)
    precision = precision_score(y_test, target_vector_y_prediction, average='binary')
    recall = recall_score(y_test, target_vector_y_prediction, average='binary')
    f1 = f1_score(y_test, target_vector_y_prediction, average='binary')
    confusion = confusion_matrix(y_test, target_vector_y_prediction).tolist()
    report = classification_report(y_test, target_vector_y_prediction, output_dict=True)

    y_pred_proba = model.predict_proba(feature_matrix_x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    feature_importance_data = calculate_permutation_importance(model, feature_matrix_x_test, y_test)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion": confusion,
        "report": report,
        "roc_auc": roc_auc,
        "feature_importance_data": feature_importance_data
    }

def calculate_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    total_importance = sum(result.importances_mean)

    if total_importance > 0:
        return [
            (feature, (importance / total_importance) * 100)
            for feature, importance in zip(X_test.columns, result.importances_mean)
        ]
    else:
        return [(feature, 0) for feature in X_test.columns]

def load_and_prepare_data(data: dict):

    label_column = data['labelColumn'].strip()
    file_path = data['file_path']

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or invalid format.")


    missing_columns = [col for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES + [label_column] if
                       col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

    try:
        numeric_features = NUMERIC_FEATURES + DATE_FEATURES
    except NameError:
        raise ValueError("DATE_FEATURES is not defined in the current scope.")

    # Preprocess price data
    clean_price_data(data)

    save_last_trained_features(CATEGORICAL_FEATURES, numeric_features, DATE_FEATURES)

    # Preprocess feature from date column
    data_frame = extract_datetime_features(data)

    feature_matrix_x = data_frame[CATEGORICAL_FEATURES + numeric_features]
    target_vector_y = data[label_column]

    return data, feature_matrix_x, target_vector_y


def clean_price_value(value):
    # Remove all characters except digits and the first decimal point
    cleaned_value = re.sub(r'[^\d.]', '', value.strip())

    # Handle cases with multiple decimal points by keeping only the first one
    parts = cleaned_value.split('.')
    if len(parts) > 2:
        cleaned_value = parts[0] + '.' + ''.join(parts[1:])

    # Convert to float
    try:
        return float(cleaned_value)
    except ValueError:
        return None  # or handle it as needed, e.g., return 0.0 or raise an exception

def clean_price_data(data):
    # Loop over each price feature in PRICE_FEATURES and apply the cleaning function
    for feature in PRICE_FEATURES:
        if feature in data.columns:
            data[feature] = data[feature].apply(
                lambda x: clean_price_value(x) if isinstance(x, str) else x
            )
    return data

def save_last_trained_features(categorical_features, numeric_features, date_features):
    # Save the features to the database
    trained_features = TrainedFeatures.objects.create(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        date_features=date_features
    )
    trained_features.save()  # Save instance to the database
    return trained_features

def load_last_trained_features():
    try:
        # Get the latest TrainedFeatures record
        last_trained_features = TrainedFeatures.objects.latest('timestamp')

        # Structure the data to be returned as JSON
        data = {
            "categorical_features": last_trained_features.categorical_features,
            "numeric_features": last_trained_features.numeric_features,
            "date_features": last_trained_features.date_features,
            "timestamp": last_trained_features.timestamp,
        }

        return JsonResponse(data)
    except TrainedFeatures.DoesNotExist:
        # If no trained features are found, return an error message
        return JsonResponse({"error": "No trained features found."}, status=404)


def extract_datetime_features(data_frame):
    data_frame["Date & Time"] = pd.to_datetime(data_frame["Date & Time"])
    data_frame["year"] = data_frame["Date & Time"].dt.year
    data_frame["month"] = data_frame["Date & Time"].dt.month
    data_frame["day"] = data_frame["Date & Time"].dt.day
    data_frame["hour"] = data_frame["Date & Time"].dt.hour
    data_frame["minute"] = data_frame["Date & Time"].dt.minute
    return data_frame.drop(columns=["Date & Time"])

# Load model if it exists
def load_model():
    if os.path.exists(MODEL_FILE_PATH):
        return joblib.load(MODEL_FILE_PATH)
    return None

# Save the model and metrics
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

def initiate_model_training(feature_matrix_x, target_vector_y):

    # Split the data into training and testing sets
    feature_matrix_x_train, feature_matrix_x_test, target_vector_y_train, target_vector_y_test = train_test_split(
        feature_matrix_x, target_vector_y, test_size=0.2, random_state=42
    )

    try:
        # Train the model with tune
        best_model, best_tune_params, test_score = model_train_with_auto_tune(X=feature_matrix_x_train, y=target_vector_y_train, categorical_features=CATEGORICAL_FEATURES, numeric_features=NUMERIC_FEATURES)
    except ValueError as e:
        logger.error("Error during model fitting: %s", e)
        logger.debug("X_train columns: %s", feature_matrix_x_train.columns.tolist())
        logger.debug("Expected categorical features: %s", CATEGORICAL_FEATURES)
        logger.debug("Expected numeric features: %s", NUMERIC_FEATURES)
        raise

    print("----------")
    print("best tuning params:")
    print(best_tune_params)
    print("----------")
    # Save the trained model
    save_model(best_model)

    # Return the model and test data for evaluation
    return best_model, target_vector_y_test, feature_matrix_x_test, best_tune_params