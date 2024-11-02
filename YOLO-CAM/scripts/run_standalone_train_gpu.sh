if [ $# != 2 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID]"
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

DEVICE_ID=0
if [ $# == 2 ]; then
    DEVICE_ID=$2
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit
echo "======start training======"
env > env.log

python train.py --data_dir=$DATASET_PATH --device_target=GPU > log.txt 2>&1 &
cd ..
