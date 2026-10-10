#!/usr/bin/env bash
set -euo pipefail
base=/validation/gdn60403-rebase-20261010
old=/validation/gdn60403-review-20261008/source
cd "$base"
test ! -e source/vllm/__init__.py
sha256sum source.tar.gz helpers.tar.gz > results/uploaded-sha256.txt
tar -xzf source.tar.gz -C source
tar -xzf helpers.tar.gz -C helpers
while IFS= read -r -d '' artifact; do
  relative=${artifact#"$old/"}
  if test ! -e "source/$relative"; then
    mkdir -p "$(dirname "source/$relative")"
    cp -P "$artifact" "source/$relative"
  fi
done < <(find "$old/vllm" -type l -name '*.so' -print0)
for relative in vllm/third_party/flashmla/flash_mla_interface.py vllm/_version.py; do
  if test -f "$old/$relative" && test ! -e "source/$relative"; then cp "$old/$relative" "source/$relative"; fi
done
bash "$base/create_container.sh"
