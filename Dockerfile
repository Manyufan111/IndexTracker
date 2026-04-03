FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOST=0.0.0.0
ENV PORT=8000

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/.data

EXPOSE 8000

CMD ["python3", "scripts/start_web.py"]
