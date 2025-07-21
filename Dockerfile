FROM python:3.9-slim-buster AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY round1a_module/ ./round1a_module/

COPY round1a_module/model/ ./round1a_module/model/

COPY persona_analyzer.py .

CMD ["python", "persona_analyzer.py"]
