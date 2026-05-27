# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: build virtual environment with all dependencies
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools, then install the project into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
COPY . .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: lean production image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
# Copy application source (no dev files, tests, or secrets)
COPY --from=builder /app/agents       ./agents
COPY --from=builder /app/api          ./api
COPY --from=builder /app/memory       ./memory
COPY --from=builder /app/models       ./models
COPY --from=builder /app/providers    ./providers
COPY --from=builder /app/config.py    ./config.py
COPY --from=builder /app/logging_config.py ./logging_config.py
COPY --from=builder /app/observability.py  ./observability.py
COPY --from=builder /app/main.py      ./main.py

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
