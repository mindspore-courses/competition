#!/bin/bash

if [ ! -d out ]; then
  mkdir out
fi
cd out || exit
if [ $MS_LITE_HOME ];then
  MINDSPORE_PATH=$MS_LITE_HOME/runtime
else
  MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  if [[ ! $MINDSPORE_PATH ]];then
      MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  fi
fi
cmake .. -DMINDSPORE_PATH=$MINDSPORE_PATH
make
