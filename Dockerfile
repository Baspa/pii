# syntax=docker/dockerfile:1.7
# Temperate PII with its model inside. Fetch the model first, then build:
#   ./fetch-model.sh
#   docker build -t temperate-pii .
#   docker run -p 8090:8090 temperate-pii
# The prebuilt ONNX Runtime that `ort` links needs glibc 2.38 or later, so both stages are Debian trixie.

FROM debian:trixie-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl pkg-config libssl-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY rust-toolchain.toml ./
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
      --default-toolchain "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)"
ENV PATH=/root/.cargo/bin:$PATH
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --locked -p pii-detection-api

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libssl3t64 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/pii-detection-api /usr/local/bin/pii-detection-api
COPY models/pii-detection-model-full-merged-int8 /opt/temperate-pii/model
ENV PII_DETECTION_ARTIFACT_DIR=/opt/temperate-pii/model \
    PII_DETECTION_ADDRESS=0.0.0.0:8090
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s CMD ["pii-detection-api", "--health-check"]
USER 65534
CMD ["pii-detection-api"]
