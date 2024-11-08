import os

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .ml import predict, retrain, tuner
import pandas as pd
import io
from django.http import JsonResponse

from .ml.models import TrainedFeatures, TrainingHistory
from .ml.retrain import retrain_model


def index(request):
    return render(request, 'index.html')

def predict(request):
    data = request.GET.dict()
    result = predict.predict(data)
    return JsonResponse({'prediction': result})


def retrain(request):
    if request.method == "POST":
        data = request.POST.dict()  # Collect form data from POST
        data_set_file = request.FILES.get('file')

        if not data_set_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Save the file to the 'uploads' directory
        fs = FileSystemStorage(location='uploads/')
        file_name = fs.save(data_set_file.name, data_set_file)
        file_path = fs.path(file_name)

        # Add the file path to the data dictionary
        data["file_path"] = file_path

        # Call the retrain function with the data
        result = retrain_model(data)

        # Optionally, remove the uploaded file after processing
        os.remove(file_path)

        return JsonResponse(result)

    return JsonResponse({"error": "Invalid request method"}, status=405)

def auto_tune(request):
    result = tuner.tune_model()
    return JsonResponse(result)


def data_set_info(request):
    if request.method == 'POST':
        new_file = request.FILES.get('file')
        if not new_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        data_frame = pd.read_csv(new_file)

        # Generate dataset summary
        buffer = io.StringIO()
        data_frame.info(buf=buffer)
        info_str = buffer.getvalue()

        column_names = data_frame.columns.tolist()

        summary = {
            "head": data_frame.head().to_html(),
            "info": info_str,
            "column_names": column_names,
            "describe": data_frame.describe(include="all").to_html(),
            "missing_values": data_frame.isnull().sum().to_dict(),
            "unique_values": data_frame.nunique().to_dict(),
        }

        return JsonResponse(summary)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

def load_last_trained_features(request):
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


def training_history(request):
    # Retrieve all training history records
    training_histories = TrainingHistory.objects.all().order_by('-timestamp')  # Order by latest first

    # Format the data as a list of dictionaries
    history_data = []
    for entry in training_histories:
        history_data.append({
            "timestamp": entry.timestamp,
            "learning_rate": entry.learning_rate,
            "max_iter": entry.max_iter,
            "max_leaf_nodes": entry.max_leaf_nodes,
            "min_samples_leaf": entry.min_samples_leaf,
            "accuracy": entry.accuracy,
            "precision": entry.precision,
            "recall": entry.recall,
            "f1_score": entry.f1_score,
            "roc_auc": entry.roc_auc,
            "confusion_matrix": entry.confusion_matrix,
            "classification_report": entry.classification_report,
            "feature_importance": entry.feature_importance
        })

    # Return the data as JSON
    return JsonResponse({"training_history": history_data}, safe=False)
