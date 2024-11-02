if [ $# != 3 ]
then
    echo "Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
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
CHECKPOINT_PATH=$(get_real_path $2)
echo $DATASET_PATH
echo $CHECKPOINT_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$3
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --data_dir=$DATASET_PATH \
    --pretrained=$CHECKPOINT_PATH > log.txt 2>&1 &
cd ..
