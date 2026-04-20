PID Control

PID balance controller that drives the Sabertooth 2x12 motor controller via PWM. Proportional term responds to pitch angle, derivative term damps pitch rate, and integral term corrects steady-state error and IMU drift. Forward/backward commands shift the balance setpoint; turning commands apply a differential wheel speed offset.
