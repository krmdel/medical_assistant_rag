FROM python:3.10-slim

WORKDIR /app

RUN pip install pipenv

COPY data/data_metadata_small.csv data/data_metadata_small.csv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --ignore-pipfile --system

COPY medical_assistant_rag .

EXPOSE 5000 8501 # For Gunicorn/Streamlit

# Use an environment variable to choose between Gunicorn and Streamlit
ENV APP_TYPE=gunicorn

CMD if [ "$APP_TYPE" = "streamlit" ]; then streamlit run app.py --server.port 8501 --server.address 0.0.0.0; else gunicorn --bind 0.0.0.0:5000 app:app; fi
