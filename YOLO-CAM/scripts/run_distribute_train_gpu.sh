if [ $# != 2 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
echo $DATASET_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

export RANK_SIZE=$2

if [ -d "distribute_train" ]; then
  rm -rf ./distribute_train
fi

mkdir ./distribute_train
cp ../*.py ./distribute_train
cp ../*.yaml ./distribute_train
cp -r ../src ./distribute_train
cp -r ../model_utils ./distribute_train
cd ./distribute_train || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py \
      --device_target=GPU \
      --data_dir=$DATASET_PATH \
      --is_distributed=1 \
      --per_batch_size=32 \
      --lr=0.025 > log.txt 2>&1 &
cd ..
