import serial

PORT = "COM8"       # le port trouvé à l'étape 2
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    ser.reset_input_buffer()
    print(f"Lecture sur {PORT} (Ctrl+C pour arrêter)\n")
    try:
        while True:
            ligne = ser.readline().decode("ascii", errors="ignore").rstrip()
            if ligne:
                print(ligne)
    except KeyboardInterrupt:
        print("\nArrêt.")