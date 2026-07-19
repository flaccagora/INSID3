FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip setuptools

WORKDIR /workspace/INSID3

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN if [ -d "CRF" ] && [ -f "CRF/setup.py" ]; then \
        cd CRF && python setup.py install; \
    fi

CMD ["python"]
