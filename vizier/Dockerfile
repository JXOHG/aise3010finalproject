
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-12

WORKDIR /trainer
COPY trainmodel-vizier.py /trainer/trainmodel.py

# Install the required packages including cloudml-hypertune
RUN pip install pandas numpy scikit-learn tensorflow \
    google-cloud-aiplatform==1.60.0 \
    google-cloud-storage \
    cloudml-hypertune

ENTRYPOINT ["python", "/trainer/trainmodel.py"]