FROM apache/airflow:2.10.4-python3.11
USER root
ARG DOCKER_CLI_VERSION=29.2.1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates tar && rm -rf /var/lib/apt/lists/* \
    # Debian's docker.io client can be too old for current Docker Desktop API versions.
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz -o /tmp/docker.tgz \
    && tar -xzf /tmp/docker.tgz -C /usr/local/bin --strip-components=1 docker/docker \
    && rm /tmp/docker.tgz \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -SL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
USER airflow
COPY requirements.txt /tmp/requirements.txt
# The base image already supplies constrained Airflow 2.10.4; project dependencies are a second install step.
RUN pip install --no-cache-dir -r /tmp/requirements.txt
