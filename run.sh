export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ~/init_cuda_shell.bash
# source activate pose-cuda-10
DEBUG="CUDA_VISIBLE_DEVICES=0 python ./train.py --rot_repr quat --use_gpu --save_path=output/quat --num_epochs1=1 --num_epochs2=1"
EXP0="CUDA_VISIBLE_DEVICES=0 python ./train.py --rot_repr quat --use_gpu --save_path=output/quat"
EXP1="CUDA_VISIBLE_DEVICES=1 python ./train.py --rot_repr mat --use_gpu --save_path=output/mat"
EXP2="CUDA_VISIBLE_DEVICES=2 python ./train.py --rot_repr bbox --use_gpu --save_path=output/bbox"
EXP3="CUDA_VISIBLE_DEVICES=3 python ./train.py --rot_repr rodr --use_gpu --save_path=output/rodr"
EXP4="CUDA_VISIBLE_DEVICES=1 python ./train.py --rot_repr euler --use_gpu --save_path=output/euler"
# tmux new -d -s exp eval "$EXP0 ; read" \; \
#     split-window -d eval "$EXP1 ; read" \; \
#     split-window -d eval "$EXP2 ; read" \; \
#     split-window -d eval "$EXP3 ; read" \; \
#     split-window -d eval "$EXP4 ; read" \; \
#     select-layout -d even-vertical \; \
#     attach \;
