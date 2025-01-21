#!/bin/bash
docker run --mount type=bind,source="$(pwd)"/src/results/,target=/workspace/src/tabular_adv_v2/results/ tab_adv_run
#sudo firefox --new-window http://127.0.0.1:8888
