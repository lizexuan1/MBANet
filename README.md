# MBANet
The official code for the paper "Investigating Intrinsic Degradation Factors by Multi-Branch Aggregation for Real-world Underwater Image Enhancement"
## Environment Preparing
python 3.6  <br>  pytorch 1.1.0
### Testing
Use the pretrained model in `./checkpoints/`. Then run the script below, the results will be saved in `./result/`

    python predict.py
    --dataroot      # The folder path of the picture you want to test
    --no_dropout 
    --name          # The checkpoint name
    --model single
    --which_direction AtoB
    --dataset_mode pair
    --which_model_netG _UNetGenerator
    --skip 1
    --self_attention
    --use_norm 1
    --use_wgan 0
    --times_residual
    --instance_norm 0
    --resize_or_crop no
    --which_epoch latest
    
    
