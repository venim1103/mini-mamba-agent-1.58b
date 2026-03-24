FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -eux; \
    if ! dpkg -s sudo >/dev/null 2>&1 \
    || ! dpkg -s ca-certificates >/dev/null 2>&1 \
    || ! dpkg -s build-essential >/dev/null 2>&1 \
    || ! dpkg -s git >/dev/null 2>&1 \
    || ! dpkg -s libcurl4-openssl-dev >/dev/null 2>&1 \
    || ! dpkg -s curl >/dev/null 2>&1 \
    || ! dpkg -s libgomp1 >/dev/null 2>&1 \
    || ! dpkg -s cmake >/dev/null 2>&1; \
    then \
        apt-get update; \
        apt-get install -y --no-install-recommends sudo ca-certificates build-essential git libcurl4-openssl-dev curl libgomp1 cmake ninja-build; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o /tmp/miniconda.sh; \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"; \
    rm -f /tmp/miniconda.sh; \
    conda config --system --set auto_update_conda false; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    conda create -n ai -y python=3.11 pip; \
    conda clean -afy; \
    existing_user_by_uid="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    existing_group_by_gid="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        if [ "$(id -u "${USERNAME}")" != "${USER_UID}" ]; then \
            usermod --uid "${USER_UID}" "${USERNAME}"; \
        fi; \
        if [ "$(id -g "${USERNAME}")" != "${USER_GID}" ]; then \
            if [ -n "${existing_group_by_gid}" ]; then \
                usermod --gid "${existing_group_by_gid}" "${USERNAME}"; \
            else \
                if ! getent group "${USERNAME}" >/dev/null 2>&1; then \
                    groupadd --gid "${USER_GID}" "${USERNAME}"; \
                else \
                    groupmod --gid "${USER_GID}" "${USERNAME}"; \
                fi; \
                usermod --gid "${USERNAME}" "${USERNAME}"; \
            fi; \
        fi; \
    elif [ -n "${existing_user_by_uid}" ]; then \
        existing_user_group="$(id -gn "${existing_user_by_uid}")"; \
        if [ "${existing_user_group}" = "${existing_user_by_uid}" ] && [ "${existing_user_group}" != "${USERNAME}" ] && ! getent group "${USERNAME}" >/dev/null 2>&1; then \
            groupmod --new-name "${USERNAME}" "${existing_user_group}"; \
        fi; \
        usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home "${existing_user_by_uid}"; \
    else \
        if [ -z "${existing_group_by_gid}" ]; then \
            groupadd --gid "${USER_GID}" "${USERNAME}"; \
        fi; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
    fi; \
    user_group="$(id -gn "${USERNAME}")"; \
    chown -R "${USERNAME}:${user_group}" "${CONDA_DIR}"; \
    printf '#!/bin/sh\nif [ -f /opt/conda/etc/profile.d/conda.sh ]; then\n    . /opt/conda/etc/profile.d/conda.sh\n    conda activate ai 2>/dev/null\nfi\n' > /etc/profile.d/conda-ai.sh; \
    chmod +x /etc/profile.d/conda-ai.sh; \
    printf '\n# Auto-activate ai conda environment\nif [ -f /opt/conda/etc/profile.d/conda.sh ]; then\n    . /opt/conda/etc/profile.d/conda.sh\n    conda activate ai >/dev/null 2>&1 || true\nfi\n' >> /etc/bash.bashrc; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

ENV CONDA_DEFAULT_ENV=ai
ENV PATH=${CONDA_DIR}/envs/ai/bin:${CONDA_DIR}/bin:${PATH}
