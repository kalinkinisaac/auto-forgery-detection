from algorithm import *
from image_tools import *

path = '/Users/isaac/Desktop/singleton/c0ck12.jpg'
orig512 = Image.open('/Users/isaac/Desktop/orig512.png' if not path else path).convert('RGB')
t6984 = self_tampered(orig512, q0=50, q1=89, box=(184, 210, 274, 320))
#save_self_tampered('/Users/isaac/Desktop/singleton/c0ck12.jpg', orig512, q0=50, q1=89, box=(184, 210, 274, 320))

def plot(image: Image):
    plt.imshow(np.array(image))
    plt.show()
