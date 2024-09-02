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
# warm up
curl 127.0.0.1:8835/models/llama2/generate \
-X POST \
-d '{"inputs":" Suggest a business idea that uses artificial intelligence","parameters":{"max_new_tokens":56, "do_sample":"False", "return_full_text":"True"}, "stream":"True"}' \
-H 'Content-Type: application/json' >/dev/null 2>&1 &
python test_serving_performance.py -X 0.1 -P 8835 -O "./" -T 5000
