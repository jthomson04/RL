#!/bin/bash
# Runs in every Ray container before Ray starts (overlay.sh adds it to SETUP_COMMAND): put this NeMo-RL's Ray into each
# image-baked Gym server venv that carries Ray, so the Gym servers can join the cluster. The image built them with the
# fork's Ray 2.56.1.
RAY_VERSION=2.58.0
export UV_CACHE_DIR=/opt/.uvc-ray UV_LINK_MODE=hardlink
n=0
bad=0
for v in /opt/gym_venvs/*/*/.venv; do
  ls -d "$v"/lib/python3.13/site-packages/ray-*.dist-info >/dev/null 2>&1 || continue
  uv pip install --no-config --quiet --python "$v/bin/python" --no-deps "ray==${RAY_VERSION}" || {
    bad=$((bad + 1))
    echo "FAIL $v"
  }
  n=$((n + 1))
done
chk=$(/opt/gym_venvs/resources_servers/code_gen/.venv/bin/python -c 'import ray; print(ray.__version__)' 2>&1 | tail -1)
echo "$(hostname): Ray ${RAY_VERSION} into ${n} Gym venvs, ${bad} failed; code_gen ray=${chk}"
