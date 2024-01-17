import context
import numpy as np
from ball import Ball
import utils as fcs

def run():
    idx = 5
    iterations, trajectory = list(context.BallTrajectories('originals').get_trajectory(idx))
    
    ball = Ball()
    table = (ball.free_falling.table.center, 
             ball.free_falling.table.direction,
             ball.free_falling.table.half_length,
             ball.free_falling.table.half_width)
    
    omega = np.array([0.0, 0.0, 0.0])
    for i in range(len(iterations)):
        t = iterations[i]*1e-6
        position = trajectory[i, :]
        if i == 0:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (trajectory[i, :]-trajectory[i-1, :])/((iterations[i]-iterations[i-1])*1e-6)
    
        ball.input_data(t, position, velocity, omega)
        if i == 50:
            trajectory_prediction = ball.prediction(ball.ball["states"], 1.0, 0.01)
    
    states_measurement = np.array(ball.ball["states_measurement"])
    states_kf = np.array(ball.ball["states_kf"])
    # fcs.plot_trajectories(states_measurement[:, 0:3], states_kf[:, 0:3], table=table)
    fcs.plot_trajectories(trajectory, trajectory_prediction, table=table)

if __name__ == '__main__':
    run()
    

            