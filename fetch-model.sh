#!/usr/bin/env bash
set -euo pipefail

# Downloads the model from Hugging Face into models/, each file checked against its digest in the manifest.
# Files that are already there and match are kept.
cd "$(dirname "${BASH_SOURCE[0]}")"

repository=temperate-works/pii
revision="${PII_MODEL_REVISION:-main}"
model=models/pii-detection-model-full-merged-int8
manifest="$model/manifest.json"

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

expected_digest() {
  local digest

  digest=$(jq -r --arg file "$1" '.files[$file]' "$manifest")
  echo "${digest#*:}"
}

for file in $(jq -r '.files | keys_unsorted[]' "$manifest"); do
  if [ -f "$model/$file" ] && [ "$(sha256 "$model/$file")" = "$(expected_digest "$file")" ]; then
    continue
  fi

  echo "fetch-model: downloading $file"
  curl -fsSL --retry 3 -o "$model/$file.part" "https://huggingface.co/$repository/resolve/$revision/$file"

  if [ "$(sha256 "$model/$file.part")" != "$(expected_digest "$file")" ]; then
    rm -f "$model/$file.part"
    echo "fetch-model: $file does not match its digest in $manifest" >&2
    exit 1
  fi

  mv "$model/$file.part" "$model/$file"
done

echo "fetch-model: the model is in $model"
