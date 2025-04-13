from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

PROJECT_ID = "aise3010finalproject"
REGION = "us-central1"
BUCKET_NAME = "gs://cloud-ai-platform-8c821022-d31d-4832-8a69-5d84538874ed" 
IMAGE_URI = f"gcr.io/{PROJECT_ID}/vizier-trainer"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

# Define hyperparameter tuning job
job = aiplatform.HyperparameterTuningJob(
    display_name="vizier_tuning_job",
    custom_job=aiplatform.CustomJob(
        display_name="vizier_training",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-4"
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "args": []
            }
        }],
    ),
    metric_spec={
        "accuracy": "maximize"
    },
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale="log"),
        "num_neurons": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale=None),
        "dropout_rate": hpt.DoubleParameterSpec(min=0.1, max=0.5, scale="linear")
    },
    max_trial_count=10,
    parallel_trial_count=3
)

# Run the job
job.run()