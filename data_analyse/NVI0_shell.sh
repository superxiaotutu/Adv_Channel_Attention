echo CUDA_VISIBLE_DEVICES 0

python VGG_data_analyse.py 8
sleep 5
python VGG_data_analyse.py 16
sleep 5
python VGG_data_analyse.py 24
sleep 5
python VGG_data_analyse.py 32
sleep 5

python ResNet50_data_analyse.py 8
sleep 5
python ResNet50_data_analyse.py 16
sleep 5
python ResNet50_data_analyse.py 24
sleep 5
python ResNet50_data_analyse.py 32
sleep 5

python InceptionV3_data_analyse.py 8
sleep 5
python InceptionV3_data_analyse.py 16
sleep 5
python InceptionV3_data_analyse.py 24
sleep 5
python InceptionV3_data_analyse.py 32
