#!/bin/bash

IMAGE_BASE_NAME="easy_deploy_base_dev"
BUILT_IMAGE_TAG=""

script_dir="$( cd "$(dirname "$0")" && pwd )"
parent_dir="$( cd "$script_dir/../.." && pwd )"
parent_dir_name="$(basename "$parent_dir")"

CONTAINER_NAME="easy_deploy_${parent_dir_name}"

usage() {
  echo "Usage: $0 --platform=<platform>"
  echo "Available platforms: jetson_trt8_u2204, jetson_trt8_u2004, nvidia_gpu, rk3588"
  exit 1
}

parse_args() {
  if [ "$#" -ne 1 ]; then
    usage
  fi
  # 解析参数
  for i in "$@"; do
      case $i in
          --platform=*)
              PLATFORM="${i#*=}"
              shift
              ;;
          *)
              usage
              ;;
      esac
  done
}

is_image_exist() {
  local name="$1"
  if docker images --filter "reference=$name" \
                   --format "{{.Repository}}:{{.Tag}}" | grep -q "$name"; then
    return 0
  else 
    return 1
  fi
}

is_container_exist() {
  local name="$1"
  if docker ps -a --filter "name=$name" | grep -q "$name"; then
    return 0
  else
    return 1
  fi
}

build_rk3588_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:rknn_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f ${script_dir}/rknn_u2204.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_jetson_trt8_u2204_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:jetson_tensorrt8_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f ${script_dir}/jetson_tensorrt_trt8_u2204.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}


build_jetson_trt8_u2004_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:jetson_tensorrt8_u2004
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f ${script_dir}/jetson_tensorrt_trt8_u2004.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_nvidia_gpu_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:nvidia_gpu_tensorrt_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f ${script_dir}/nvidia_gpu_tensorrt_u2204.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_image() {
  case $PLATFORM in
      jetson_trt8_u2204)
          echo "Start Building Docker image for Jetson TensorRT8 Ubuntu 2204 platform..."
          build_jetson_trt8_u2204_image
          ;;
      jetson_trt8_u2004)
          echo "Start Building Docker image for Jetson TensorRT8 Ubuntu 2004 platform..."
          build_jetson_trt8_u2004_image
          ;;
      nvidia_gpu)
          echo "Start Building Docker image for nvidia_gpu platform..."
          build_nvidia_gpu_image
          ;;
      rk3588)
          echo "Start Building Docker image for rk3588 platform..."
          build_rk3588_image
          ;;
      *)
          echo "Unknown platform: $PLATFORM"
          usage
          ;;
  esac
}

add_user() {
  echo Adding User: ${USER} into container
   
}

create_container() {
  echo "Creating docker container ..."

  if ! is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} does not exist, quit creating ...
    exit 1
  fi

  if is_container_exist ${CONTAINER_NAME}; then
    echo Container: ${CONTAINER_NAME} exists! Skip container building process ...
    return 0
  fi

  EXTERNAL_TAG=""
  case $PLATFORM in
      jetson_trt8_u2204)
          EXTERNAL_TAG="--runtime nvidia"
          ;;
      jetson_trt8_u2004)
          EXTERNAL_TAG="--runtime nvidia"
          ;;
      nvidia_gpu)
          EXTERNAL_TAG="--runtime nvidia"
          ;;
      rk3588)
          
          ;;
      *)
          ;;
  esac

  docker run -itd --privileged \
             --device /dev/dri \
             --group-add video \
             -v /tmp/.X11-unix:/tmp/.X11-unix \
             --network host \
             --ipc host \
             -v ${parent_dir}:/workspace \
             -w /workspace \
             -v /dev/bus/usb:/dev/bus/usb \
             -e DISPLAY=${DISPLAY} \
             -e DOCKER_USER=${USER} \
             -e USER=${USER} \
             --name ${CONTAINER_NAME} \
             ${EXTERNAL_TAG} \
             ${BUILT_IMAGE_TAG} \
             /bin/bash
}

parse_args "$@"

build_image

create_container

echo "EasyDeploy Base Dev Enviroment Built Successfully!!!"
echo "Now Run into_docker.sh"
