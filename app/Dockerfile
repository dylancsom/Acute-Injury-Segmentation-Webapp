FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app/woundseg-flask-app

COPY requirements.txt app.py run_app.py june11v9.keras ./
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501

ENV MODEL_PATH june11v9.keras

CMD streamlit run --server.port $PORT --server.address 0.0.0.0 app.py
