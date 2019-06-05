# auto-forgery-detection
Automated Image Forgery Detection through Classification of JPEG Ghosts

### How to start

1. In console `git clone https://github.com/kalinkinisaac/auto-forgery-detection.git`
2. Set up your virtual enviroment (python3)
3. Install packages `pip install -r requirements.txt`
4. Go to the folder `cd auto-forgery-detection/`
5. You are ready to go! You are perfect!

### How to chech photo

1. Open python console, when you are in project folder `python`
2. Import all from managy.py `from manage import *`
3. Let be you photo you want to analise be `path = /Users/user/Downloads/test.jpg`
4. Determine path to photo in console `path = 'path'`
5. Run prediction script `predict_img(path, w=64)`
6. You will see your image, and the prediction process will start.
7. If your image analisys is not accurate, than you should decrease value of `w`. 
For example here you can see difference `w=64` to the right, and `w=16` to the left:
[![Example 1](https://a.radikal.ru/a10/1906/40/1718c4fde245.png "Example 1")](https://a.radikal.ru/a10/1906/40/1718c4fde245.png "Example 1")
8. After you guessed **the right** `w=w0`, than you are able to predict finally.

Green and Red squares (named *blocks*) represents difference between to images.

### Used sources

- [Article: 'Automated Image Forgery Detection through Classification'](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Zach12-AIF.pdf "Automated Image Forgery Detection through Classification")
- [Documentation of scikit-learn 0.21.2](https://scikit-learn.org/stable/documentation.html "Documentation of scikit-learn 0.21.2")
- [NumPy User Guide](https://docs.scipy.org/doc/numpy/user/index.html#user "NumPy User Guide")
