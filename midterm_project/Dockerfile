# Use official slim Python 3.12 image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Upgrade pip and install pipenv
RUN pip install --upgrade pip pipenv

# Copy dependency files first for caching
COPY ["Pipfile","Pipfile.lock","./"]

# Install dependencies into system Python (no virtualenv inside container)
RUN pipenv install --system --deploy

# Copy application code and model
COPY predict.py predict_flask.py model_depth_10_lr_0.02_iter_200.bin ./

EXPOSE 9898

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9898","predict_flask:app" ]
