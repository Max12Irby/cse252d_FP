#!/bin/bash

cd ./cse252d_FP 
./restart_gpu.bash

# if restarted correctly
if test $? -eq 0
then
  sleep 2
  echo "Starting detr/main.py"
  if (test $# -gt 0) && (test $1 == "encoder")
  then
      echo "Start detr using swin as backbone+encoder"
      ./kubesh-nostdin-TEMP.bash raltai-gpu -- bash cse252d_FP/detr_swin_encoder.sh &
  else
      echo "Start detr using swin as backbone"
      ./kubesh-nostdin-TEMP.bash raltai-gpu -- bash cse252d_FP/detr_swin_backbone.sh &
  fi
else 
  echo "Failed to restart pod!"
  exit 1
fi
