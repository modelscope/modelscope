docker build \
    --network host \
    -f Dockerfile.metax \
    -t swift:v4.2.3 \
    --build-arg VLLM_VERSION=v0.20.0 \
    --build-arg VLLM_METAX_VERSION=v0.20.0 \
    --build-arg MACA_VERSION=3.7.0 \
    --build-arg MEGATRON_VERSION=core_v0.16.0 \
    --build-arg SWIFT_VERSION=v4.2.3 \
    --build-arg TE_VERSION=2.8.0+4a002bf5.maca3.5.3.105.torch2.8 \
    --build-arg CU_BRIDGE_VERSION=3.7.0 \
    .
