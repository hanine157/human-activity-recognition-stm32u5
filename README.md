# Human Activity Recognition on STM32U5

Embedded human activity recognition using the ISM330DHCX accelerometer on the
B-U585I-IOT02A board. The firmware acquires acceleration over I2C, collects
26 samples on the data-ready interrupt, normalizes the three axes, and runs a
three-class neural network generated with STM32Cube.AI.

## Activities

The embedded labels are:

- `stationary`
- `walking`
- `running`

The WISDM preparation pipeline maps `Sitting` and `Standing` to `stationary`,
`Walking` to `walking`, and `Jogging` to `running`. Windows contain 26 samples
and three axes (`x`, `y`, `z`), matching the generated model input shape
`1 x 26 x 3`.

## Repository contents

- `Core/`, `Drivers/`, `Middlewares/`, `X-CUBE-AI/`: STM32CubeIDE firmware.
- `STM32LAB1/`: UART monitoring and plotting tools.
- `ml/src/`: WISDM loading, preprocessing, CNN, and Random Forest scripts.

The raw WISDM dataset is not copied into this repository. Download it separately
and keep its `readme.txt` and attribution when redistributing it.

## Python pipeline

Create an environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Prepare windows from the raw WISDM file:

```powershell
py ml\src\prepare_data.py C:\path\to\WISDM_ar_v1.1_raw.txt
```

Train the baselines:

```powershell
py ml\src\train_random_forest.py ml\data\wisdm_windows.npz
py ml\src\train_cnn.py ml\data\wisdm_windows.npz
```

The split is performed by user, so windows from the same person do not appear
in both training and test sets.

## Firmware

Open the project in STM32CubeIDE, build it, flash the B-U585I-IOT02A board, and
connect the UART configured by the project at 115200 baud. The firmware prints
initialization messages and, for each complete window, the three output scores
and the predicted activity.

`STM32LAB1/tracer.py` currently plots raw `x,y,z` CSV lines. The current
firmware prints inference messages rather than raw CSV samples, so the plotting
tool requires a raw-sample UART mode before it can be used with this firmware.

## Citation

The WISDM dataset requests citation of:

J. R. Kwapisz, G. M. Weiss, and S. A. Moore, “Activity Recognition using Cell
Phone Accelerometers,” Proceedings of KDD-10, 2010.
