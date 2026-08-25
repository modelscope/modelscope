docker build \
    --network host \
    -f Dockerfile.metax \
    -t swift:v4.4.0 \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg VLLM_VERSION=v0.22.0 \
    --build-arg VLLM_METAX_VERSION=v0.22.0 \
    --build-arg MACA_VERSION=3.8.0 \
    --build-arg MEGATRON_VERSION=core_v0.16.0 \
    --build-arg SWIFT_VERSION=v4.4.0 \
    --build-arg TE_VERSION=2.13.0 \
    --build-arg CU_BRIDGE_VERSION=3.8.0 \
    .
