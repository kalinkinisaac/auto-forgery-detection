from algorithm import *
from image_tools import *

path = 'data/c0ck12.jpg'
orig512 = Image.open('/Users/isaac/Desktop/orig512.png' if not path else path).convert('RGB')
t6984 = self_tampered(orig512, q0=50, q1=89, box=(800, 800, 1200, 1200))


# save_self_tampered('singleton/f8k1fuk32.jpg', t6984, q0=50, q1=89, box=(184, 210, 274, 320))

# data/f8k1fuk32.jpg -- fake    data/zhx7p.png -- true
# data/c0ck12.jpg -- fake

def open_img(p=path):
    return Image.open(p).convert('RGB')


def predict_img(p=path, w=64):
    img = Image.open(p).convert('RGB')
    print('\t %%%%%% \t plotting inputed image \t %%%%%% \t')
    plot(img)
    print('\t %%%%%% \t starting analisys process \t %%%%%% \t')
    predict(img, w)
    print('\t %%%%%% \t done \t %%%%%% \t')


def plot(image: Image):
    plt.imshow(np.array(image))
    plt.show()
