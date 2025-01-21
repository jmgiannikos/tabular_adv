#!/bin/bash
docker run --rm -p 8888:8888 --mount type=bind,source="$(pwd)",target=/workspace/ --name dev_env -d tab_adv_dev
docker exec -it dev_env git config --global --add safe.directory /workspace
docker exec wandb login --relogin 6ef4d4a363d1fa1fe106a6d9c34af35a3c3eac12

#sudo firefox --new-window http://127.0.0.1:8888
