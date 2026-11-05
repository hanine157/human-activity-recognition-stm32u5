import serial
from collections import deque
import matplotlib.pyplot as plt

PORT = "COM8"
BAUD_RATE = 115200
WINDOW = 100

ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
ser.reset_input_buffer()

plt.ion()
fig, ax = plt.subplots(num="Acquisition STM32 en Temps Réel")
ax.set_xlabel("Échantillons")
ax.set_ylabel("Accélération (mg)")
ax.set_title("Accéléromètre ISM330DHCX - Temps Réel")
ax.grid(True)

tdata = deque(maxlen=WINDOW)
xdata = deque(maxlen=WINDOW)
ydata = deque(maxlen=WINDOW)
zdata = deque(maxlen=WINDOW)

(lx,) = ax.plot([], [], "r-", label="Axe X")
(ly,) = ax.plot([], [], "g-", label="Axe Y")
(lz,) = ax.plot([], [], "b-", label="Axe Z")
ax.legend(loc="upper right")

print("Lecture lancée. Fermez la fenêtre du graphique pour arrêter.")

t = 0
try:
    while plt.fignum_exists(fig.number):
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                plt.pause(0.001)
                continue
            valeurs = [int(v) for v in line.split(",")]
            if len(valeurs) != 3:
                continue
        except ValueError:
            continue  # ignore "UART OK", "Sensor found", etc.

        t += 1
        tdata.append(t)
        xdata.append(valeurs[0])
        ydata.append(valeurs[1])
        zdata.append(valeurs[2])

        lx.set_data(tdata, xdata)
        ly.set_data(tdata, ydata)
        lz.set_data(tdata, zdata)

        ax.set_xlim(max(1, t - WINDOW), max(WINDOW, t))
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        plt.pause(0.001)
finally:
    ser.close()
    print("Acquisition arrêtée.")