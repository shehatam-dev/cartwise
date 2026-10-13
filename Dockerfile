FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
RUN pip install --no-cache-dir ".[gemini,otlp]"

# docker run -it cartwise                      (offline scripted model)
# docker run -it -e CARTWISE_PROVIDER=gemini -e GEMINI_API_KEY=... cartwise
ENTRYPOINT ["cartwise"]
CMD ["chat"]
