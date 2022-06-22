import cv2
from matplotlib import pyplot as plt


def cv_show(img, title=None, save=False):
    if save:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(title + '.png', img, dpi=600)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
