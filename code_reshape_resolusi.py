import sys
import numpy as np
import os

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from tensorflow.keras.models import load_model
import threading

manipulation_frame = None
label = ''
hasil = ''
hasil_confidence = ''

class appleThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = load_model("model/Apple.hdf5")
        while (~(manipulation_frame is None)):
            label = self.predict(manipulation_frame)

    def predict(self, manipulation_frame):
        global hasil
        global hasil_confidence

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        image = cv2.cvtColor(manipulation_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        image = image.reshape((1,) + image.shape)
        preds = self.model.predict(image)
        # Apple
        index = {'Almost Ripe': 0, 'Ripe': 1, 'Underdone': 2}

        confidence_array = preds[0]
        index_max = np.argmax(confidence_array)

        category_names = index.keys()
        category_values = index.values()
        category_at_index = list(category_values).index(index_max)
        category_max = list(category_names)[category_at_index]

        hasil = (f"{category_max}")
        hasil_confidence = (f"{max(confidence_array) * 100:.2f}%")
        # return hasil, hasil_confidence

class mangoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = load_model(
            "model/Mango.hdf5")
        while (~(manipulation_frame is None)):
            label = self.predict(manipulation_frame)

    def predict(self, manipulation_frame):
        global hasil
        global hasil_confidence

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        image = cv2.cvtColor(manipulation_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        image = image.reshape((1,) + image.shape)
        preds = self.model.predict(image)
        # Apple
        index = {'Almost Ripe': 0, 'Partially Ripe': 1, 'Ripe': 2, 'Underdone': 3}

        confidence_array = preds[0]
        index_max = np.argmax(confidence_array)

        category_names = index.keys()
        category_values = index.values()
        category_at_index = list(category_values).index(index_max)
        category_max = list(category_names)[category_at_index]

        hasil = (f"{category_max}")
        hasil_confidence = (f"{max(confidence_array) * 100:.2f}%")


class orangeThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = load_model(
            "model/Orange.hdf5")
        while (~(manipulation_frame is None)):
            label = self.predict(manipulation_frame)

    def predict(self, manipulation_frame):
        global hasil
        global hasil_confidence

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        image = cv2.cvtColor(manipulation_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        image = image.reshape((1,) + image.shape)
        preds = self.model.predict(image)
        # Apple
        index = {'Almost Ripe': 0, 'Partially Ripe': 1, 'Ripe': 2, 'Unripe': 3}

        confidence_array = preds[0]
        index_max = np.argmax(confidence_array)

        category_names = index.keys()
        category_values = index.values()
        category_at_index = list(category_values).index(index_max)
        category_max = list(category_names)[category_at_index]

        hasil = (f"{category_max}")
        hasil_confidence = (f"{max(confidence_array) * 100:.2f}%")


class tomatoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = load_model(
            "model/Tomato.hdf5")
        while (~(manipulation_frame is None)):
            label = self.predict(manipulation_frame)

    def predict(self, manipulation_frame):
        global hasil
        global hasil_confidence

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        image = cv2.cvtColor(manipulation_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255
        image = image.reshape((1,) + image.shape)
        preds = self.model.predict(image)
        # Apple
        index = {'Almost Ripe': 0, 'Partially Ripe': 1, 'Ripe': 2, 'Unripe': 3}

        confidence_array = preds[0]
        index_max = np.argmax(confidence_array)

        category_names = index.keys()
        category_values = index.values()
        category_at_index = list(category_values).index(index_max)
        category_max = list(category_names)[category_at_index]

        hasil = (f"{category_max}")
        hasil_confidence = (f"{max(confidence_array) * 100:.2f}%")


class PREDICT(QMainWindow):
    def __init__(self):
        super(PREDICT, self).__init__()
        loadUi("app.ui", self)

        self.logic = 0
        self.value = 1
        self.predict_apple_btn.clicked.connect(self.predict_apple_onClicked)
        self.predict_mango_btn.clicked.connect(self.predict_mango_onClicked)
        self.predict_orange_btn.clicked.connect(self.predict_orange_onClicked)
        self.predict_tomato_btn.clicked.connect(self.predict_tomato_onClicked)
        self.TEXT.setText("Select One!")
        # self.TEXT.setStyleSheet("background-color: red; border: 1px solid black;")
        self.TEXT.setStyleSheet("color: red;")
        # self.CAPTURE.clicked.connect(self.CaptureClicked)
        self.CAPTURE.setVisible(False)
        self.TEXT_3.setVisible(False)
        # self.exit_btn.clicked.connect(self.exit)
        self.exit_btn.setVisible(False)
        self.stop_apple_btn.setVisible(False)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', ' Are you sure you want to close window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            print('Window closed')
            sys.exit()
        else:
            event.ignore()

    # def exit(self):
    #     if self.close():
    #         sys.exit()


    @pyqtSlot()
    def predict_apple_onClicked(self):
        global manipulation_frame
        global hasil
        global hasil_confidence

        self.TEXT.setText('')
        self.TEXT_3.setText('Capture for screenshoot')
        self.TEXT_3.setStyleSheet("color: red;")
        self.stop_apple_btn.setVisible(True)

        self.cap = cv2.VideoCapture(0)
        if (self.cap.isOpened()):
            # print("Camera Apple OK")
            self.lbl_camera_status.setText('Camera Apple OK')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            self.timer = QtCore.QTimer()
        else:
            self.cap.open()

        keras_thread = appleThread()
        keras_thread.start()

        while (True):
            self.ret, self.frame = self.cap.read()
            manipulation_frame = cv2.resize(self.frame, (224, 224))

            self.displayImage(self.frame, 1)
            self.label_hasil_klasifikasi.setText(hasil)
            self.label_hasil_confidence.setText(hasil_confidence)
            # self.displayImage(cv2.putText(frame, "Result: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2), 1)
            cv2.waitKey()

    ##fitur screenshoot
            # if (self.logic == 2):
            #     self.value = self.value + 1
            #     cv2.imwrite('Capture/%s.png' % (self.value), frame)
            #     self.logic = 1
            #     self.TEXT.setText('your Image have been Saved')
            # else:
            #     print('')

    @pyqtSlot()
    def predict_mango_onClicked(self):

        global manipulation_frame
        global hasil
        global hasil_confidence

        self.TEXT.setText('')
        self.TEXT_3.setText('Capture for screenshoot')
        self.TEXT_3.setStyleSheet("color: red;")
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()):
            # print("Camera Apple OK")
            self.lbl_camera_status.setText('Camera Mango OK')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        else:
            cap.open()

        keras_thread = mangoThread()
        keras_thread.start()

        while (True):
            ret, frame = cap.read()
            manipulation_frame = cv2.resize(frame, (224, 224))

            self.displayImage(frame, 1)
            self.label_hasil_klasifikasi.setText(hasil)
            self.label_hasil_confidence.setText(hasil_confidence)
            cv2.waitKey()

            # #fitur screenshoot
            # if (self.logic == 2):
            #     self.value = self.value + 1
            #     cv2.imwrite('Capture/%s.png' % (self.value), frame)
            #     self.logic = 1
            #     self.TEXT.setText('your Image have been Saved')
            # else:
            #     print('')

        cap.release()
        manipulation_frame = None
        cv2.destroyAllWindows()

    @pyqtSlot()
    def predict_orange_onClicked(self):

        global manipulation_frame
        global hasil
        global hasil_confidence

        self.TEXT.setText('')
        self.TEXT_3.setText('Capture for screenshoot')
        self.TEXT_3.setStyleSheet("color: red;")
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()):
            self.lbl_camera_status.setText('Camera Orange OK')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        else:
            cap.open()

        keras_thread = orangeThread()
        keras_thread.start()

        while (True):
            ret, frame = cap.read()
            manipulation_frame = cv2.resize(frame, (224, 224))

            self.displayImage(frame, 1)
            self.label_hasil_klasifikasi.setText(hasil)
            self.label_hasil_confidence.setText(hasil_confidence)
            cv2.waitKey()

            # # fitur screenshoot
            # if (self.logic == 2):
            #     self.value = self.value + 1
            #     cv2.imwrite('Capture/%s.png' % (self.value), frame)
            #     self.logic = 1
            #     self.TEXT.setText('your Image have been Saved')
            # else:
            #     print('')

        cap.release()
        manipulation_frame = None
        cv2.destroyAllWindows()

    @pyqtSlot()
    def predict_tomato_onClicked(self):

        global manipulation_frame
        global hasil
        global hasil_confidence

        self.TEXT.setText('')
        self.TEXT_3.setText('Capture for screenshoot')
        self.TEXT_3.setStyleSheet("color: red;")
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()):
            self.lbl_camera_status.setText('Camera Tomato OK')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        else:
            cap.open()

        keras_thread = tomatoThread()
        keras_thread.start()

        while (True):
            ret, frame = cap.read()
            manipulation_frame = cv2.resize(frame, (224, 224))

            self.displayImage(frame, 1)
            self.label_hasil_klasifikasi.setText(hasil)
            self.label_hasil_confidence.setText(hasil_confidence)
            cv2.waitKey()

            # # fitur screenshoot
            # if (self.logic == 2):
            #     self.value = self.value + 1
            #     cv2.imwrite('Capture/%s.png' % (self.value), frame)
            #     self.logic = 1
            #     self.TEXT.setText('your Image have been Saved')
            # else:
            #     print('')

        cap.release()
        manipulation_frame = None
        cv2.destroyAllWindows()

    def CaptureClicked(self):
        self.logic = 2

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


app = QApplication(sys.argv)
window = PREDICT()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('excitng')
