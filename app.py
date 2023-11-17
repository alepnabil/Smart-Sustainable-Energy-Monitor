import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from pyfirmata import Arduino, util


class YoloHumanDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(0, 0, 1920, 1080)  # Set the size based on your laptop screen resolution

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.power_plot = pg.PlotWidget(self)
        self.power_plot.setLabel('right', 'Power Consumption (kwH)', size=10000, color='white')
        self.power_plot.setLabel('top', 'Time', size=10000, color='white')
        self.power_plot.plotItem.setSizePolicy(pg.QtWidgets.QSizePolicy.Expanding, pg.QtWidgets.QSizePolicy.Expanding)
        self.power_curve = self.power_plot.plot(pen='r')

        self.cost_plot = pg.PlotWidget(self)
        self.cost_plot.setLabel('right', 'Cost (Ringgit)', size=10000, color='white')
        self.cost_plot.setLabel('top', 'Time', size=10000, color='white')
        self.cost_plot.plotItem.setSizePolicy(pg.QtWidgets.QSizePolicy.Expanding, pg.QtWidgets.QSizePolicy.Expanding)
        self.cost_curve = self.cost_plot.plot(pen='g', symbol='o')

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(self.video_label)
        central_layout.addWidget(self.power_plot)
        central_layout.addWidget(self.cost_plot)

        self.setCentralWidget(central_widget)

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("Error: Could not open camera.")
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000)  # Set the desired frame rate (milliseconds)

        self.time_elapsed = 0
        self.power_data = []
        self.cost_data = []
        self.lower_range = 1
        self.upper_range = 10

        self.arduino_port = 'COM10'  # Replace with the correct COM port of your Arduino
        # Create a PyFirmata board object
        self.board = Arduino(self.arduino_port)
        # Create an iterator to read data from the board
        self.it = util.Iterator(self.board)
        self.it.start()
        self.lightbulb= self.board.get_pin('d:3:o')

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            return

        frame, human_detected = self.detect_human(frame)
        self.display_frame(frame)

        # Set minimum size for the video label after loading the video frame
        self.video_label.setMinimumSize(frame.shape[1], frame.shape[0])

        # Update power consumption and cost data only if a human is detected
        if human_detected:
            self.time_elapsed += 1
            power_value = np.random.uniform(10, 5000)  # Replace this with actual power consumption data
            self.lower_range = self.lower_range + 100
            self.upper_range = self.upper_range + 2000
            cost_values = self.generate_nonlinear_cost()[-1]  # Use the last value in the generated cost data
            self.power_data.append((self.time_elapsed, power_value))
            self.cost_data.append((self.time_elapsed, cost_values))
            self.update_power_plot()
            self.update_cost_plot()


    def detect_human(self, frame):
        # Initialize human_detected
        human_detected = False

        # YOLO detection logic
        net = cv2.dnn.readNet("D:\\tnb_smart_tariff_system\\weight\\yolov3-tiny (1).weights",
                              "D:\\tnb_smart_tariff_system\\cfg\\yolov3-tiny.cfg")

        # Load coco names
        classes = []
        with open("D:\\tnb_smart_tariff_system\\lib\\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getUnconnectedOutLayersNames()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2 and class_id == 0:  # Check if it's a human (class_id 0)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    human_detected = True
                    class_ids.append(class_id)



        if human_detected:
            self.lightbulb.write(1)
        else:
            self.lightbulb.write(0)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        return frame, human_detected

    def display_frame(self, frame):
        # Convert color space from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def generate_nonlinear_cost(self):
        # Function to generate non-linear increasing cost values
        base_cost = 50  # Adjust the base cost to set the minimum cost value
        increment = np.random.uniform(self.lower_range, self.upper_range)  # Adjust the increment range
        return base_cost + np.cumsum(increment)

    def update_power_plot(self):
        # Update power consumption plot
        x, y = zip(*self.power_data)
        self.power_curve.setData(x=x, y=y)

    def update_cost_plot(self):
        # Update cost plot
        x, y = zip(*self.cost_data)
        self.cost_curve.setData(x=x, y=y)

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = YoloHumanDetectionApp()
    window.show()
    app.exec_()
