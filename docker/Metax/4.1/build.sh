docker build \
    --network host \
    -f Dockerfile.metax \
    -t swift:v4.1.0 \
    --build-arg VLLM_VERSION=v0.17.1 \
    --build-arg VLLM_METAX_VERSION=v0.17.0 \
    --build-arg MACA_VERSION=3.5.3 \
    --build-arg MEGATRON_VERSION=core_v0.16.0 \
    --build-arg SWIFT_VERSION=v4.1.0 \
    --build-arg TE_VERSION=2.8.0 \
    --build-arg CU_BRIDGE_VERSION=3.5.3 \
    --no-cache \
    .
