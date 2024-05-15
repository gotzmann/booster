# sudo docker build -t zoo
# sudo docker run -it --rm --runtime=nvidia --gpus all -v /home/models:/home/models -p 8080:8080 zoo

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

COPY collider collider
COPY config.yaml config.yaml

ENTRYPOINT ["./booster", "--debug", "--server"]