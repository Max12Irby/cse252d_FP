# cse252d_FP

Final project for CSE 252d. We will be modifying the DETR object detection framework by changing the encoder (currently ViT) and backbone (currently CNN). Will test performance of: Swin transformer, ConViT, and ViT. Will change out out different combinations of encoder-only, backbone-only, and both merged together.


### Commands

You can start training after starting a gpu node from dsmlp-login by running this command. You won't have to stay ssh'ed, and output will be saved to a log file.
```
# For replacing backbone with Swin
./kubesh-nostdin-TEMP.bash <Name of GPU node> -- bash cse252d_FP/detr_swin_backbone.sh & 
# For replacing backbone + encoder with Swin
./kubesh-nostdin-TEMP.bash <Name of GPU node> -- bash cse252d_FP/detr_swin_encoder.sh &
```
