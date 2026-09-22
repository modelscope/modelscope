docker build \
    --network host \
    -f Dockerfile.with_metax_image \
    -t swift:v4.4.0-with-metax-base \
    --build-arg VLLM_VERSION=v0.22.0 \
    --build-arg VLLM_METAX_VERSION=v0.22.0 \
    --build-arg MEGATRON_VERSION=core_v0.16.0 \
    --build-arg SWIFT_VERSION=v4.4.0 \
    --build-arg TE_VERSION=2.13.0 \
    .
