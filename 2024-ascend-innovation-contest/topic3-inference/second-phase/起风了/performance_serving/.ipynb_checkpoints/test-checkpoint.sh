#/bin/bash

#python test_serving_performance.py -X 2.0
# python test_serving_performance.py -X 2.2
# python test_serving_performance.py -X 2.4
# python test_serving_performance.py -X 2.6
# python test_serving_performance.py -X 2.8
# python test_serving_performance.py -X 3.0
# python test_serving_performance.py -X 3.2
# python test_serving_performance.py -X 3.4
# python test_serving_performance.py -X 3.6

python test_serving_performance.py -X 1 -P 8835 -O "./" -T 5
