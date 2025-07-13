FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY score.py .

# install all runtime deps, including scikit-learn
RUN pip install click pandas numpy pyarrow scikit-learn

ENTRYPOINT ["python", "score.py"]
