FROM python:3.11-slim

# Install uv (fast package manager — replaces pip inside Docker)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (Docker layer caching — only re-installs on change)
COPY pyproject.toml requirements.txt ./

# Install production dependencies only (no dev/test tools in the image)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
