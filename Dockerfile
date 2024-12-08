FROM python:3.11-slim


RUN mkdir /app
WORKDIR /app
COPY pyproject.toml /app
COPY main.py /app
RUN pip install . --no-cache

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
