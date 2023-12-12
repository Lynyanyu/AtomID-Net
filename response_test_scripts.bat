python ./response_test.py --device cuda --method NMS --crop_size 800
python ./response_test.py --device cpu --method NMS --crop_size 800
python ./response_test.py --device cuda --method LoG --crop_size 800
python ./response_test.py --device cuda --method DoG --crop_size 800
python ./response_test.py --device cuda --method DoH --crop_size 800
pause