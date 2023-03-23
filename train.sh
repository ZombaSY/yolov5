CUDA_VISIBLE_DEVICES=2 python3 train.py --img 512 --batch 128 --epochs 3000 --patience 300 --data hair_detection.yaml --weights pretrained/yolov5s.pt
