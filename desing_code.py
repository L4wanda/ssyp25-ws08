import sys
import numpy as np
from PyQt6.QtWidgets import *
import matplotlib.animation as an
from PyQt6.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skydome_cuda import SkydomeRenderer
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from datetime import datetime
import requests
import suncalc
import math


class ExampleApp(QMainWindow):
    def __init__(self):

        QMainWindow.__init__(self)

        loadUi('testswin.ui', self)
        self.fig_rel = Figure()
        self.canvas_rel = FigureCanvas(self.fig_rel)
        self.canvas_rel.axes_rel = self.fig_rel.add_subplot()

        self.width = self.resolution_spinbox.value()
        self.height = self.width
        self.resolution_spinbox.valueChanged.connect(self.resolution_update)

        self.scale_height_rayleigh = self.spinBox_2.value()
        self.spinBox_2.valueChanged.connect(self.height_rayleigh_update)

        t = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.im_rayleigh = self.canvas_rel.axes_rel.imshow(t, )
        self.canvas_rel.axes_rel.set_title("Rayleigh")
        self.canvas_rel.axes_rel.axis("off")

        layout_rel = QVBoxLayout(self.rel_graph)  # rel_graph - название виджета, который Вы создали в Qt Designer
        layout_rel.addWidget(self.canvas_rel)

        self.pushButton.clicked.connect(self.toggle_rayleigh)

        self.city = None
        self.latitude = None
        self.longitude = None
        self.anim_rayleigh = None
        self.anim_mie = None
        self.anim_sum = None
        self._playing = False       # своё состояние

        self.fig_mie = Figure()
        self.canvas_mie = FigureCanvas(self.fig_mie)
        self.canvas_mie.axes_mie = self.fig_mie.add_subplot()

        self.im_mie = self.canvas_mie.axes_mie.imshow(t)
        self.canvas_mie.axes_mie.set_title("Mie")
        self.canvas_mie.axes_mie.axis("off")

        layout = QVBoxLayout(self.mi_graph)  # mi_graph - название виджета, который Вы создали в Qt Designer
        layout.addWidget(self.canvas_mie)

        self.pushButton_2.clicked.connect(self.toggle_mie)

        self.fig_sum = Figure()
        self.canvas_sum = FigureCanvas(self.fig_sum)
        self.canvas_sum.axes_sum = self.fig_sum.add_subplot()

        self.im_sum = self.canvas_sum.axes_sum.imshow(t)
        self.canvas_sum.axes_sum.set_title("Rayleigh + Mie")
        self.canvas_sum.axes_sum.axis("off")

        layout_sum = QVBoxLayout(self.Sum_graph)  # Sum_graph - название виджета, который Вы создали в Qt Designer
        layout_sum.addWidget(self.canvas_sum)

        self.GenSum_graph.clicked.connect(self.toggle_sum)

        self.fig_mie.tight_layout()
        self.fig_rel.tight_layout()
        self.fig_sum.tight_layout()
        self.city_text.editingFinished.connect(self.city_updated)

        self.pushButton_3.clicked.connect(self.save_rayleigh_gif)
        self.pushButton_4.clicked.connect(self.save_mie_gif)
        self.pushButton_5.clicked.connect(self.save_sum_gif)


    def city_updated(self):
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.GenSum_graph.setEnabled(True)
        _geolocator = Nominatim(user_agent="city_timezone")
        self.city = self.city_text.text()
        _location = _geolocator.geocode(self.city)
        self.latitude = _location.latitude
        self.longitude = _location.longitude
        azimuth, altitude, humidity, b, native_time = self.weather_info()
        if azimuth == -1 and altitude == -1 and humidity == -1 and b == -1:
            self.info_label.setText(f'Wrong city')
        else:
            self.info_label.setText(f'Humidity: {humidity}%, Azimuth: {azimuth:.2f}°, Altitude: {altitude:.2f}°, Time: {native_time}')
            self.spinBox_3.setValue(b)


    def update_rayleigh(self, frame):
        minutes = (frame * 10) % (24 * 60) # every 10 minutes
        color = SkydomeRenderer(False, self.latitude, self.longitude,
                                self.width, self.height, self.scale_height_rayleigh).render(minutes)
        color = np.array(color)
        color[color>1] = 1
        cur_hours = minutes // 60
        cur_minutes = minutes % 60
        self.canvas_rel.axes_rel.set_title(f"Rayleigh,   Time = {cur_hours:02d}:{cur_minutes:02d}")
        self.im_rayleigh.set_data(color)


    def update_mie(self, frame):
        minutes = (frame * 10) % (24 * 60) # every 10 minutes
        color = SkydomeRenderer(True, self.latitude, self.longitude,
                                self.width, self.height, self.scale_height_rayleigh).render(minutes)
        color = np.array(color)
        color[color>1] = 1
        cur_hours = minutes // 60
        cur_minutes = minutes % 60
        self.canvas_mie.axes_mie.set_title(f"Mie,    Time = {cur_hours:02d}:{cur_minutes:02d}")
        self.im_mie.set_data(color)


    def update_sum(self, frame):
        minutes = (frame * 10) % (24 * 60) # every 10 minutes
        color = (SkydomeRenderer(False, self.latitude, self.longitude,
                                 self.width, self.height, self.scale_height_rayleigh).render(minutes)
                 + SkydomeRenderer(True, self.latitude, self.longitude,
                                   self.width, self.height, self.scale_height_rayleigh).render(minutes))
        color = np.array(color)
        color[color>1] = 1
        cur_hours = minutes // 60
        cur_minutes = minutes % 60
        self.canvas_sum.axes_sum.set_title(f"Mie + Rayleigh,   Time = {cur_hours:02d}:{cur_minutes:02d}")
        self.im_sum.set_data(color)


    def toggle_sum(self, checked: bool):
        if self.anim_sum is None:
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(False)
            self.anim_sum = an.FuncAnimation(self.fig_sum, self.update_sum, frames=24 * 60, interval=50)
            self.canvas_sum.draw()
            self.GenSum_graph.setText("⏸ Pause")
            self.GenSum_graph.setChecked(True)
            return

        if checked:
            self.anim_sum.resume()
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(False)
            self.GenSum_graph.setText("⏸ Pause")
        else:
            self.anim_sum.pause()
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)
            self.GenSum_graph.setText("▶ Play")
        self.canvas_sum.draw()

    def toggle_rayleigh(self, checked: bool):
        if self.anim_rayleigh is None:
            self.pushButton_2.setEnabled(False)
            self.GenSum_graph.setEnabled(False)
            self.anim_rayleigh = an.FuncAnimation(self.fig_rel, self.update_rayleigh, frames=24 * 60, interval=50)
            self.canvas_rel.draw()
            self.pushButton.setText("⏸ Pause")
            self.pushButton.setChecked(True)
            return

        if checked:
            self.anim_rayleigh.resume()
            self.pushButton_2.setEnabled(False)
            self.GenSum_graph.setEnabled(False)
            self.pushButton.setText("⏸ Pause")
        else:
            self.anim_rayleigh.pause()
            self.pushButton_2.setEnabled(True)
            self.GenSum_graph.setEnabled(True)
            self.pushButton.setText("▶ Play")

    def toggle_mie(self, checked: bool):
        if self.anim_mie is None:
            self.pushButton.setEnabled(False)
            self.GenSum_graph.setEnabled(False)
            self.anim_mie = an.FuncAnimation(self.fig_mie, self.update_mie, frames=24 * 60, interval=50)
            self.canvas_mie.draw()
            self.pushButton_2.setText("⏸ Pause")
            self.pushButton_2.setChecked(True)
            return

        if checked:
            self.anim_mie.resume()
            self.pushButton.setEnabled(False)
            self.GenSum_graph.setEnabled(False)
            self.pushButton_2.setText("⏸ Pause")
        else:
            self.anim_mie.pause()
            self.pushButton.setEnabled(True)
            self.GenSum_graph.setEnabled(True)
            self.pushButton_2.setText("▶ Play")


    def weather_info(self):
        try:
            geolocator = Nominatim(user_agent="city_timezone")
            location = geolocator.geocode(self.city)
            if not location:
                return -1, -1, -1, -1, -1
            latitude = location.latitude
            longitude = location.longitude

            #print(f"Широта : {latitude}")
            #print(f"Долгота : {longitude}")

            tf = TimezoneFinder()
            iana_timezone = tf.timezone_at(lat=latitude, lng=longitude)
            if not iana_timezone:
                return -1, -1, -1, -1, -1
            tz = ZoneInfo(iana_timezone)
            utc_offset_seconds = tz.utcoffset(datetime.now()).total_seconds()
            gmt_offset_hours = utc_offset_seconds / 3600

            #print(f"Часовой пояс: {iana_timezone}")
            #print(f"Смещение от UTC : {gmt_offset_hours}")

        except ValueError as e:
            return -1, -1, -1, -1, -1

        city_name = self.city
        API_KEY = 'f1b0714cc430d8f574e7598567e859d9'
        base_url = f'https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={API_KEY}&units=metric'

        try:
            url = f"{base_url}"
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.json()

            humidity = weather_data['list'][0]['main']['humidity']
            b = 21e-6 / 50 * humidity
            #print(f"Влажность: {humidity}%")
            #print('betaM', b)

            dt_txt = weather_data['list'][0]['dt_txt']
            year = int(dt_txt[0:4])
            month = int(dt_txt[5:7])
            day = int(dt_txt[8:10])
            hour_1 = int(dt_txt[11:13])
            minute = int(dt_txt[14:16])
            second = 0

            hour_utc = hour_1 + gmt_offset_hours
            if hour_utc > 23:
                hour_utc -= 24
            elif hour_utc < 0:
                hour_utc += 24
            hour = int(hour_utc)

            naive_datetime = datetime(year, month, day, hour, minute, second)
            city_timezone = ZoneInfo(iana_timezone)
            aware_datetime = naive_datetime.replace(tzinfo=city_timezone)
            utc_datetime = aware_datetime.astimezone(ZoneInfo('UTC'))

            # print(f"Местное время: {naive_datetime}")
            # print(f"Местное время: {aware_datetime}")
            # print(f"Время в UTC: {utc_datetime}")

            sun_pos = suncalc.get_position(utc_datetime, longitude, latitude)
            altitude = math.degrees(sun_pos['altitude'])
            azimuth = 180 + math.degrees(sun_pos['azimuth'])

            # print(f"Азимут: {azimuth:.2f} градусов")
            # print(f"Высота: {altitude:.2f} градусов")
            # print(sun_pos)  #Вывод sun_pos
            # print(weather_data)
            # print(sun_pos)
            # return weather_data
            return azimuth, altitude, humidity, b, hour
        except requests.exceptions.RequestException as e:
            return -1, -1, -1, -1, -1

        except Exception as e:
            return -1, -1, -1, -1, -1


    def save_rayleigh_gif(self):
        if self.anim_rayleigh is None:
            QMessageBox.warning(self, "Нет анимации",
                                "Сначала запустите расчёт Rayleigh.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить GIF",
                                              "", "GIF (*.gif)")
        if path:
            self.anim_rayleigh.save(path, writer=an.PillowWriter(fps=24), dpi=150)

    def save_mie_gif(self):
        if self.anim_mie is None:
            QMessageBox.warning(self, "Нет анимации",
                                "Сначала запустите расчёт Mie.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить GIF",
                                              "", "GIF (*.gif)")
        if path:
            self.anim_mie.save(path, writer=an.PillowWriter(fps=24), dpi=150)

    def save_sum_gif(self):
        if self.anim_sum is None:
            QMessageBox.warning(self, "Нет анимации", "Сначала запустите расчёт Mie.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить GIF","", "GIF (*.gif)")
        if path:
            self.anim_sum.save(path, writer=an.PillowWriter(fps=24), dpi=150)

    def resolution_update(self, value):
        self.width = value
        self.height = value

    def height_rayleigh_update(self, value):
        self.scale_height_rayleigh = value



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    app.setPalette(QApplication.style().standardPalette())
    window = ExampleApp()
    window.show()
    app.exec()