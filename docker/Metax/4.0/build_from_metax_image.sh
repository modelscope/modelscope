docker build \
    --network host \
    -f Dockerfile.with_metax_image \
    -t swift:v4.0.0 \
    --build-arg VLLM_VERSION=v0.11.2 \
    --build-arg VLLM_METAX_VERSION=v0.11.2 \
    --build-arg MEGATRON_VERSION=core_v0.15.0  \
    --build-arg SWIFT_VERSION=v4.0.0 \
    --progress=plain \
    --no-cache \
     .



