FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev,web]"

COPY . .

# Default entrypoint — override with --entrypoint for specific commands:
#   rhyme-generate  Generate corpus + query set
#   rhyme-run       Run baselines or custom adapter
#   rhyme-score     Score results
#   rhyme-probe     Run adversarial style probe
ENTRYPOINT ["python", "-m"]
