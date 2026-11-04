FROM ghcr.io/positive666/model@sha256:ea31ef16a4f389f30f6ec7651814e4a3aeda7fa54b2dc012e43323313b6965fb
COPY context/ /opt/image-lmcache-lifecycle-20261104-r1/
RUN --mount=type=bind,source=.,target=/build,readonly \
    /opt/glm53-image-gn/bin/uv pip install --python /opt/glm53-image-gn/.venv/bin/python --no-deps --offline --reinstall /build/wheels/lmcache-*.whl && \
    /opt/glm53-image-gn/bin/uv run --offline --no-project --python /opt/glm53-image-gn/.venv/bin/python /opt/glm53-image-gn/.venv/bin/python -B /opt/image-lmcache-lifecycle-20261104-r1/install_release.py
LABEL ai.glm53.provenance.path="/opt/image-lmcache-lifecycle-20261104-r1/SOURCE_LOCK.json" \
      ai.glm53.validation="see-external-acceptance-receipt" \
      ai.glm53.release.status="release-candidate" \
      ai.glm53.lmcache-revision="ca439cad3ea983ae8dc6197379070f6b71bfee61" \
      ai.glm53.lmcache.version="0.5.7.dev267+glm53.ca439cad.life1" \
      ai.glm53.update-kind="rebuilt-lmcache-lifecycle-fixed-parent-runtime" \
      io.positive666.release="20261104-lifecycle-r1"
