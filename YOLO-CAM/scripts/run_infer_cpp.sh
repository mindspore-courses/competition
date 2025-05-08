if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_TYPE] [DEVICE_ID]
    DVPP is mandatory, and must choose from [DVPP|CPU], it's case-insensitive
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
ann_file=$(get_real_path $3)
DVPP=${4^^}

device_id=0
if [ $# == 6 ]; then
    device_id=$6
fi

if [ $5 == 'GPU' ]; then
    device_id=0
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "ann file: "$ann_file
echo "image process mode: "$DVPP
echo "device id: "$device_id

if [ $5 == 'Ascend' ] || [ $5 == 'GPU' ] || [ $5 == 'CPU' ]; then
  device_type=$5
else
  echo "DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
  exit 1
fi
echo "device type: "$device_type

if [ $MS_LITE_HOME ]; then
  RUNTIME_HOME=$MS_LITE_HOME/runtime
  TOOLS_HOME=$MS_LITE_HOME/tools
  RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
  RUNTIME_LIBS=$RUNTIME_LIBS:$RUNTIME_HOME/third_party/dnnl/
  export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
  echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
fi


function compile_app()
{
    cd ../cpp_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    if [ "$DVPP" == "DVPP" ];then
      echo "Only support CPU mode"
      exit 1
    elif [ "$DVPP" == "CPU"  ]; then
      ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --dataset_path=$data_path --device_id=$device_id --image_height=640 --image_width=640 &> infer.log
    else
      echo "image process mode must be in [DVPP|CPU]"
      exit 1
    fi
}

function cal_acc()
{
    python ../postprocess.py --result_files=./result_Files --dataset_path=$data_path --ann_file=$ann_file &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi