#!/bin/bash
docker run --rm -p 8888:8888 --mount type=bind,source="$(pwd)"/src/,target=/workspace/src/ tabular_adv_v2
#sudo firefox --new-window http://127.0.0.1:8888
