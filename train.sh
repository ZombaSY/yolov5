CUDA_VISIBLE_DEVICES=0,1 python train.py --img 512 --batch 32 --epochs 3000 --patience 500 --data hair_detection.yaml --weights pretrained/yolov5s.pt
