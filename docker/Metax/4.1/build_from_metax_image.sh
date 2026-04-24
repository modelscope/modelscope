docker build \
    --network host \
    -f Dockerfile.with_metax_image \
    -t swift:v4.1.0 \
    --build-arg VLLM_VERSION=v0.17.1 \
    --build-arg VLLM_METAX_VERSION=v0.17.0 \
    --build-arg MEGATRON_VERSION=core_v0.16.0 \
    --build-arg SWIFT_VERSION=v4.1.0 \
    --no-cache \
    .
