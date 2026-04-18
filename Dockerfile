FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev,web]"

COPY . .

# Default entrypoint — override with --entrypoint for specific commands:
#   sifter-generate  Generate corpus + query set
#   sifter-run       Run baselines or custom adapter
#   sifter-score     Score results
#   sifter-probe     Run adversarial style probe
ENTRYPOINT ["python", "-m"]
