rm -R /sciclone/home20/hmbaier/tflow/ips/
mkdir /sciclone/home20/hmbaier/tflow/ips/

rm -R /sciclone/home20/hmbaier/tflow/logs/
mkdir /sciclone/home20/hmbaier/tflow/logs/

for ((i = 1; i <= $1; i++))
do
	qsub /sciclone/home20/hmbaier/tflow/tflow_job.sh -l nodes=1:$2:ppn=1 -v NODE_NUM=$i,WORLD_SIZE=$1
done