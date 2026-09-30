FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[ml,app]'
COPY configs ./configs
COPY data ./data
COPY artifacts ./artifacts

EXPOSE 8501
CMD ["streamlit", "run", "src/bikeflow/dashboard.py", "--server.address=0.0.0.0"]

