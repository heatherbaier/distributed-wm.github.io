#!/bin/tcsh
#PBS -N head_node
#PBS -l walltime=10:00:00
#PBS -j oe


hostname -i > /sciclone/home20/hmbaier/tflow/ips/$NODE_NUM.txt

set size=`ls /sciclone/home20/hmbaier/tflow/ips/ | wc -l`

while ( $size != $WORLD_SIZE )
    set size=`ls /sciclone/home20/hmbaier/tflow/ips/ | wc -l`
    sleep 1
end

echo "$size"


# init conda within new shell for job
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
unsetenv PYTHONPATH
conda activate tflow

python3 /sciclone/home20/hmbaier/tflow/worker_v5.py $NODE_NUM $WORLD_SIZE > "/sciclone/home20/hmbaier/tflow/logs/log${NODE_NUM}.txt"