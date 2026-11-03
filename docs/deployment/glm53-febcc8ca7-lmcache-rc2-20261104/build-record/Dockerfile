FROM ghcr.io/positive666/model@sha256:8f4d9c6e79eec0444a88b855b4af05be631bfcf3ca93ec625b9c4380b5ea0fe0
COPY context/ /opt/image-main-realign-20261103-r3/
RUN --mount=type=bind,source=.,target=/build,readonly \
    printf '%s\n' 'nvidia-nccl-cu13==2.31.2' > /etc/uv-overrides.txt && \
    UV_OVERRIDE=/dev/null /opt/glm53-image-gn/bin/uv pip install --python /opt/glm53-image-gn/.venv/bin/python --target /usr/local/lib/python3.12/dist-packages --no-deps --offline --reinstall /build/inputs/nvidia_nccl_cu13-2.31.2-*.whl && \
    /opt/glm53-image-gn/bin/uv pip install --python /opt/glm53-image-gn/.venv/bin/python --no-deps --offline --reinstall /build/inputs/vllm-*.whl /build/wheels/lmcache-*.whl /build/humming-wheels/*.whl /build/deepep-wheels/*.whl && \
    /opt/glm53-image-gn/bin/uv run --offline --no-project /opt/glm53-image-gn/.venv/bin/python -B /opt/image-main-realign-20261103-r3/install_release.py
LABEL org.opencontainers.image.revision="febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc" \
      org.opencontainers.image.version="0.33.0rc1.dev607+gfebcc8ca7" \
      ai.glm53.vllm-native-revision="febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc" \
      ai.glm53.source.commit="febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc" \
      ai.glm53.provenance.path="/opt/image-main-realign-20261103-r3/SOURCE_LOCK.json" \
      ai.glm53.validation="see-external-acceptance-receipt" \
      ai.glm53.release.status="release-candidate" \
      ai.glm53.vllm="febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc" \
      ai.vllm.build.commit="febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc" \
      ai.glm53.lmcache-revision="386473dad97fdad4112a7333bc6604185acfd084" \
      ai.glm53.lmcache.version="0.5.7.dev264+glm53.386473dad.align3" \
      ai.glm53.humming-revision="145bed14896b34b35bd6934dc35063975b47ee28" \
      ai.glm53.humming-version="0.1.19.post1+g145bed148" \
      ai.glm53.nccl-version="2.31.2" \
      ai.glm53.deepep-revision="d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb" \
      ai.glm53.update-kind="official-vllm-wheel-rebuilt-lmcache-humming-deepep-nccl-alignment" \
      io.positive666.release="20261103-r3"
