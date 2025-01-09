FROM python:3.11-slim

RUN mkdir /app
RUN mkdir /app/src
WORKDIR /app


COPY src /app/src
COPY pyproject.toml /app
COPY entrypoint.sh /app
RUN chmod +x /app/entrypoint.sh
RUN pip install . --no-cache 


ENTRYPOINT ["/app/entrypoint.sh"]
