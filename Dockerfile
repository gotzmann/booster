FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENTRYPOINT ["echo $NV_CUDA_LIB_VERSION"]