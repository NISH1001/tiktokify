FROM python:3.12-slim

# Install system dependencies for Playwright/Chromium + curl for uv
RUN apt-get update && apt-get install -y \
    curl \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy project files for uv sync
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies with uv sync (uses lockfile)
RUN uv sync --frozen --no-dev --extra web

# Set Playwright browser path to shared location accessible by non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright

# Install Playwright browsers to shared path
RUN uv run playwright install chromium

# Copy remaining application code
COPY app.py .

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Set environment variables
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright

EXPOSE 7860

CMD ["uv", "run", "python", "app.py"]
