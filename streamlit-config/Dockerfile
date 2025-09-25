FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --no-cache-dir -r requirements.txt 

RUN python -m playwright install-deps

RUN python -m playwright install chromium

EXPOSE 8501

COPY streamlit-config/secrets.toml .streamlit/secrets.toml

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]