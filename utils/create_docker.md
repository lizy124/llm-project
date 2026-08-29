docker run -dit -u root \
-p 0.0.0.0:8181:22 \
--name refactor_818 \
-e ASCEND_RUNTIME_OPTIONS=NODRV \
--privileged=true \
-v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver \
-v /home:/home \
-v /data:/data \
-v /tmp:/tmp \
-v /mnt:/mnt \
-v /usr/local/sbin/:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--shm-size=100g \
--net=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-w /home \
9a24d007653b \
/bin/bash