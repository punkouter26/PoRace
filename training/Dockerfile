# PoRace trainer image: JAX CUDA 13 + MuJoCo Warp + mujoco_playground + Brax.
# Build:  docker build -t porace-trainer training
# Run:    docker run --rm -it --gpus all -v "%cd%/training:/work" -p 6006:6006 porace-trainer bash
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/venv UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv git libgl1 libegl1 libosmesa6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/
WORKDIR /work
COPY pyproject.toml ./
RUN uv sync --no-install-project --python 3.12
# mujoco_playground clones menagerie on first import; bake it in so training runs offline and pinned.
RUN M=$(/venv/bin/python -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_PATH)") && S=$(/venv/bin/python -c "from mujoco_playground._src import mjx_env; print(mjx_env.MENAGERIE_COMMIT_SHA)") \n    && git clone -q https://github.com/google-deepmind/mujoco_menagerie.git "$M" && git -C "$M" checkout -q "$S" && rm -rf "$M/.git"
ENV PATH=/venv/bin:$PATH MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD ["bash"]
