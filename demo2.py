import sys
import matplotlib.image as mpimg
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

NOMEROFF_NET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, "Mask_RCNN/")
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")

sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet import Detector, filters, RectDetector

rectDetector = RectDetector()

nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

nnet.loadModel("latest")

print("START RECOGNIZING")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('des.ui', self)
        self.title = 'Тест'
        self.left = 30
        self.top = 60
        self.width = 40
        self.height = 50
        self.filename = None
        self.pixmap = None
        self.outputfile = None
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # self.pixmap = QPixmap('12345.png')
        # self.pixmap4 = self.pixmap.scaled(600, 300, Qt.KeepAspectRatio)
        # self.im1.setPixmap(self.pixmap4)
        # self.im1.resize(self.pixmap4.width(), self.pixmap4.height())
        #
        # self.resize(self.pixmap.width(), self.pixmap.height())
        # self.show()

        self.load.clicked.connect(self.getFileName)

        self.plainTextEdit = QPlainTextEdit()
        self.plainTextEdit.setFixedSize(100, 100)
        self.plainTextEdit.setFont(QFont('Arial', 1))

        self.start.clicked.connect(self.bluring)

        self.resize(800, 880)
        self.setWindowTitle("Проектище")
        self.show()

    def getFileName(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Выбрать файл",
                                                         ".",
                                                         "JPEG Files(*.jpeg);;\
                                                         PNG Files(*.png);;")
        self.plainTextEdit.appendHtml("Выбрали файл: <b>{}</b>"
                                      "".format(filename, filetype))
        self.filename = filename
        pixmap = QPixmap(filename)

        pixmap4 = pixmap.scaled(600, 300, Qt.KeepAspectRatio)
        self.im1.setPixmap(pixmap4)
        self.im1.resize(pixmap4.width(), pixmap4.height())

        self.pathw.setText(self.filename)
        # self.pathw.adjustSize()
        self.pathw.setFont(QFont("Times", 10))

    def bluring(self):
        def blurContours(image, contours, ksize, sigmaX, *args):
            sigmaY = args[0] if len(args) > 0 else sigmaX
            mask = np.zeros(image.shape[:2])
            for i, contour in enumerate(contours):
                cv2.drawContours(mask, contour, i, 255, -1)
            blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX, None, sigmaY)
            result = np.copy(image)
            alpha = mask / 255
            result = alpha[:, :, None] * blurred_image + (1 - alpha)[:, :, None] * result
            return result

        ind = self.filename.rfind('/')
        path = self.filename[:ind]
        file = self.filename[ind + 1:]
        outputfile = path + '/blurphoto/' + file
        self.outputfile = outputfile

        if not os.path.isdir(path + '/blurphoto/'):
            os.mkdir(path + '/blurphoto/')

        img = mpimg.imread(self.filename)
        NP = nnet.detect([img])
        cv_img_masks = filters.cv_img_mask(NP)
        res = []
        arrPoints = rectDetector.detect(cv_img_masks)
        intListArrPoint = arrPoints.astype(int).tolist()
        bluredImg = cv2.imread(self.filename)
        for i in intListArrPoint:
            bluredImg = blurContours(bluredImg, [[np.array(i)]], 25, 12, 12)
        cv2.imwrite(outputfile, bluredImg)

        image = Image.open(outputfile)

        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)

        image_without_exif.save(outputfile)

        pixmap = QPixmap(outputfile)

        pixmap4 = pixmap.scaled(600, 300, Qt.KeepAspectRatio)
        self.im2.setPixmap(pixmap4)
        self.im2.resize(pixmap4.width(), pixmap4.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
