import cv2, numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

def cv_to_qimage(img):
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        qimg = QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format_BGR888)
    return qimg

class ImageView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400,300)
        self.image = None
        self._blink_imgs = None
        self._blink = False
        self._blink_idx = 0
        self._timer = QtCore.QTimer(self, interval=500)
        self._timer.timeout.connect(self._on_blink_tick)

        self._fade_imgs = None
        self._fade_alpha = 0.5

    def set_image(self, img):
        self.image = img
        self.update()

    def set_blink_pair(self, imgA, imgB):
        self._blink_imgs = (imgA, imgB)
        self._blink_idx = 0

    def set_fade_pair(self, imgA, imgB):
        self._fade_imgs = (imgA, imgB)
        self._fade_alpha = 0.5

    def start_blink(self, on=True, interval_ms=500):
        self._blink = on
        if on and self._blink_imgs:
            self._timer.setInterval(interval_ms)
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def set_fade_alpha(self, a):
        self._fade_alpha = np.clip(a, 0.0, 1.0)
        self.update()

    def _on_blink_tick(self):
        if not self._blink_imgs: return
        self._blink_idx = 1 - self._blink_idx
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(20,20,20))

        img = self.image
        if self._blink and self._blink_imgs:
            img = self._blink_imgs[self._blink_idx]

        if self._fade_imgs is not None:
            A, B = self._fade_imgs
            if A is not None and B is not None:
                # alpha blend in painter
                qA = cv_to_qimage(A)
                qB = cv_to_qimage(B)
                target = self.rect()
                p.setOpacity(1.0)
                p.drawImage(target, qA)
                p.setOpacity(self._fade_alpha)
                p.drawImage(target, qB)
                p.setOpacity(1.0)
                p.end()
                return

        if img is not None:
            q = cv_to_qimage(img)
            p.drawImage(self.rect(), q)
        p.end()
