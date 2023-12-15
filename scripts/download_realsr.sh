#!/bin/bash
directory=./data/RealSR
file="./data/RealSR/RealSR(V3).tar.gz"
mkdir -p "${directory}"
gdown -O "${file}" --id 17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM
tar -xzf "${file}" -C "${directory}" && rm "${file}"
