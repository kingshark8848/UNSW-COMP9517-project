## Running Environment
python 3.4+ 
OpenCV 3.1+

## File Structure & Intro

### colorAnalyze.py
system use it to analyze color features of training image data (table/balls)

command line:
~~~
python3 colorAnalyze.py -m hsv -i test_data/game1/frame76.png
python3 colorAnalyze.py -m hsv -i test_data/game2/5_sample.png
~~~

### imutils.py
some util functions.

### config.py 
basic configuration for running program.

### video.py:
includes all classes/functions used in video process (ball detection/tracking)

### sim.py
includes all classes/functions used in simulation

### projection.py
includes all functions used in projection

### run.py
run the program
command line:
~~~
python3 run.py
~~~
