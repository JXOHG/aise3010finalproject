from google.cloud import aiplatform
aiplatform.init(
    project='aise3010finalproject',
    location='us-central1'  # e.g., 'us-central1'
)
model = aiplatform.Model('7398355358072176640')
evaluations = model.list_model_evaluations()


for eval in evaluations:
    print(dict(eval.metrics))  # Convert to regular Python dictionary
