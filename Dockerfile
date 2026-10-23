FROM ubuntu:24.04

ARG RUST_VERSION=1.93.1
ARG MRACLE_BUILD_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/workspace/target/release:/usr/local/cargo/bin:$PATH
ENV CC=clang-21
ENV CXX=clang++-21
ENV MRACLE_BUILD_JOBS=${MRACLE_BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        lsb-release \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-venv \
        pkg-config \
        libatomic1 \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
        > /etc/apt/sources.list.d/llvm-21.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-21 \
        lld-21 \
        libclang-rt-21-dev \
        libomp-21-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rustup component add rustfmt clippy

WORKDIR /workspace
COPY . .

RUN cargo build --locked --release

CMD ["mracle-fuzz", "--help"]
