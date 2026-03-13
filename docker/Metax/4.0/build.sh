docker build \
    --network host \
    -f Dockerfile.metax \
    -t swift:v4.0.0 \
    --build-arg VLLM_VERSION=v0.11.2 \
    --build-arg VLLM_METAX_VERSION=v0.11.2 \
    --build-arg MACA_VERSION=3.3.0 \
    --build-arg MEGATRON_VERSION=core_v0.15.0  \
    --build-arg SWIFT_VERSION=v4.0.0 \
    --build-arg TE_VERSION=2.8 \
    --build-arg CU_BRIDGE_VERSION=3.3.0 \
    --no-cache \
    .
