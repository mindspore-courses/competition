model_path=../data/models/yolov5.air
output_model_name=../data/models/yolov5

atc --framework=1 \
    --model="${model_path}" \
    --input_shape="actual_input_1:1,12,320,320"  \
    --output="${output_model_name}" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision \
    --output_type=FP32
exit 0
