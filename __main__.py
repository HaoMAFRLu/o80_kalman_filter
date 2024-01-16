import context
import numpy as np
from ball import Ball

if __name__ == '__main__':
    idx = 10
    iterations, trajectory = list(context.BallTrajectories('originals').get_trajectory(idx))
    ball = Ball()
    omega = np.array([0.0, 0.0, 0.0])
    for i in range(len(iterations)):
        t = iterations[i]*1e-6
        position = trajectory[i, :]
        if i == 0:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (trajectory[i, :]-trajectory[i-1, :])/((iterations[i]-iterations[i-1])*1e-6)
        ball.input_data(t, position, velocity, omega)

            