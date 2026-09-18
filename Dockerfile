ARG PYTORCH_IMAGE=10.43.250.50/black-tea-simple/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MPLBACKEND=Agg \
    HOME=/home/app

WORKDIR /app

# The base image supplies the CUDA-matched torch and torchvision builds.
# This is an immutable application image layer, so allow pip to add the
# remaining dependencies to the base image's PEP 668-managed Python.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app \
    && mkdir -p /data/input /data/runs \
    && chown -R "${APP_UID}:${APP_GID}" /app /data /home/app

COPY --chown=${APP_UID}:${APP_GID} . /app

USER ${APP_UID}:${APP_GID}

# Fail the image build early if the application or its declared runtime
# dependencies cannot be imported. This command does not need a dataset/GPU.
RUN python tools/train.py --help >/dev/null

# Kubernetes Jobs override CMD with a concrete config while retaining Python
# as the executable, for example: tools/train.py --config configs/example.yaml
ENTRYPOINT ["python"]
CMD ["tools/train.py", "--help"]
