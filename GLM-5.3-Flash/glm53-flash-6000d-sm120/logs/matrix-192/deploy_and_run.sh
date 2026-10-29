#!/bin/bash
# .192: wait for DSV4.1 k357+nospec matrix to finish, then deploy GLM env+weights from .155 and run k5+nospec
OPS=/root/glm_matrix
LOG=$OPS/deploy.log
mkdir -p "$OPS"
{
echo "chain armed $(date '+%F %T')"
} > "$LOG"

# wait for the last DSV4.1 leg (nospec) to complete
for i in $(seq 1 3000); do
  [ -f /root/test_ops/k357_192/nospec_done ] && break
  sleep 30
done
if [ ! -f /root/test_ops/k357_192/nospec_done ]; then
  echo "wait for nospec_done timed out" >> "$LOG"; exit 1
fi

{
sleep 60
date '+%F %T DSV4.1 matrix finished, GLM deploy start'
echo "== rsync fixbenv =="
rsync -a root@<source-host>:/root/fixbenv/ /root/fixbenv/
echo "venv RC=$?"
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python3
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python3.12
# .192 has an incomplete uv-managed cpython-3.12 (missing _ctypes); the venv's
# pyvenv.cfg inherited from .155 points at it and would shadow the system stdlib.
sed -i 's|^home = .*|home = /usr/bin|' /root/fixbenv/pyvenv.cfg
/root/fixbenv/bin/python -c 'import vllm, flashinfer, transformers, torch; print("ver", vllm.__version__, flashinfer.__version__, transformers.__version__, torch.__version__)'
RCV=$?
echo "vercheck RC=$RCV"
[ "$RCV" -ne 0 ] && { echo DEPLOY_FAIL; touch $OPS/DEPLOY_FAIL; exit 1; }

echo "== rsync weights 306G =="
rsync -a root@<source-host>:/data/models/GLM-5.3-Flash/ /data/models/GLM-5.3-Flash/
echo "weights RC=$?"
NS=$(ls /data/models/GLM-5.3-Flash/model-*.safetensors 2>/dev/null | wc -l)
echo "shards=$NS"
[ "$NS" != "62" ] && { echo DEPLOY_FAIL; touch $OPS/DEPLOY_FAIL; exit 1; }
a=$(md5sum /data/models/GLM-5.3-Flash/model-00001-of-00062.safetensors | cut -d' ' -f1)
[ "$a" = "967e2f9f44e869935627f4b02e965706" ] && echo "md5 ok" || { echo "MD5 MISMATCH"; touch $OPS/DEPLOY_FAIL; exit 1; }
date '+%F %T deploy done'
touch $OPS/DEPLOY_DONE
} >> "$LOG" 2>&1

if [ -f $OPS/DEPLOY_DONE ]; then
  cd $OPS && setsid nohup bash glm_matrix.sh </dev/null >/dev/null 2>&1 &
  echo "matrix launched $(date)" >> "$LOG"
fi
