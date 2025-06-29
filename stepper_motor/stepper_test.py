from time import sleep
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

# Define GPIO pins
direction = 20            # Direction GPIO pin
step = 21                 # Step GPIO pin
# MS1, MS2, MS3
mode_pins = (14, 15, 18)

# Set GPIO mode - RpiMotorLib handles this, but it's good practice
GPIO.setmode(GPIO.BCM)

# --- FIX for GPIOConverter Bug ---
# The compatibility layer cannot handle setting up a list of pins.
# We set them up one by one manually BEFORE initializing the motor.
for pin in mode_pins:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(direction, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)
# --------------------------------

# Create motor instance with correct pin order
# RpiMotorLib.A4988Nema(direction_pin, step_pin, mode_pins, motor_type)
motor = RpiMotorLib.A4988Nema(direction, step, mode_pins, "A4988")

# Run motor: rotate 200 steps clockwise, with 0.005s delay between steps
motor.motor_go(
    clockwise=True,      # Clockwise direction
    steptype="Full",     # Step type: Full, Half, 1/4, 1/8, 1/16
    steps=200,           # Number of steps (200 = 1 full revolution for 1.8° motor)
    stepdelay=0.005,     # Delay between steps
    verbose=True,        # Print status to console
    initdelay=0.05       # Delay before starting
)

# No need to call GPIO.cleanup() if you let the RpiMotorLib object go out of scope,
# but it's good practice if the script continues.
# The library's destructor should handle it.
sleep(1)

print("Rotating counter-clockwise...")
motor.motor_go(
    clockwise=False,
    steptype="Full",
    steps=200,
    stepdelay=0.005,
    verbose=True,
    initdelay=0.05
)


GPIO.cleanup()
