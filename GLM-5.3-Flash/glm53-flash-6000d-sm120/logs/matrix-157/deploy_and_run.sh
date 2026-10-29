#!/bin/bash
# .157: pull GLM-5.3-Flash weights + fixbenv from .155, fix symlinks, verify, launch k5+nospec matrix
OPS=/root/glm_matrix
LOG=$OPS/deploy.log
mkdir -p "$OPS"
{
date '+%F %T deploy start'
echo "== rsync fixbenv =="
rsync -a root@<source-host>:/root/fixbenv/ /root/fixbenv/
echo "venv RC=$?"
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python3
ln -sf /usr/bin/python3.12 /root/fixbenv/bin/python3.12
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
# spot md5
a=$(md5sum /data/models/GLM-5.3-Flash/model-00001-of-00062.safetensors | cut -d' ' -f1)
[ "$a" = "967e2f9f44e869935627f4b02e965706" ] && echo "md5 ok" || { echo "MD5 MISMATCH"; touch $OPS/DEPLOY_FAIL; exit 1; }
date '+%F %T deploy done'
touch $OPS/DEPLOY_DONE
} > "$LOG" 2>&1

# only launch matrix after deploy success
if [ -f $OPS/DEPLOY_DONE ]; then
  cd $OPS && setsid nohup bash glm_matrix.sh </dev/null >/dev/null 2>&1 &
  echo "matrix launched $(date)" >> "$LOG"
fi
