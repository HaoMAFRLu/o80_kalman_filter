"""Classes and functions for applying Kalman filter
to estimate the states (position, velocity and spin) 
of an object
"""

import numpy as np

import utils as fcs
from mytypes import Matrix, State3d, State6d, State9d
from free_falling import FreeFalling

def get_diag_matrix(*arrs) -> Matrix:
    """This function is used to generate 
    a diagonal matrix using the given arrays

    Parameters
    ----------
    arrs:
      the given arrays
    
    Returns
    -------
    A diagonal matrix
    """
    merged_array = fcs.merge_arrays(*arrs)
    diag_matrix = np.diag(merged_array)
    return diag_matrix


class KalmanFilter:
    """ Instances of this object will be used to apply
    kalman filter to estimate the states of a ball

    Parameters
    ----------
    ini_omega:
      the the initial spin of the object, in rad/s
    sigma_r:
      the process standard deviation of the positon
    sigma_v:
      the process standard deviation of the velocity
    sigma_omega:
      the process standard deviation of the spin
    sigma_y:
      the process standard deviation of the output
    ini_sigma_r: 
      the initial standard deviation of the position
    ini_sigma_v: 
      the initial standard deviation of the velocity
    ini_sigma_omega:
      the initial standard deviation of the spin
    """
    def __init__(self, config) -> None:
        self.ini_omega = config["ini_omega"]
        
        self.sigma_r = np.array(config["sigma"]["r"])
        self.sigma_v = np.array(config["sigma"]["v"])
        self.sigma_y = np.array(config["sigma"]["y"])
        self.sigma_omega = np.array(config["sigma"]["omega"])

        self.ini_sigma_r = np.array(config["ini_sigma"]["r"])
        self.ini_sigma_v = np.array(config["ini_sigma"]["v"])
        self.ini_sigma_omega = np.array(config["ini_sigma"]["omega"])

    def reset(self) -> None:
        """This function is used to reset and initialize
        the kalman filter

        Parameters
        ----------
        Q:
          process noise covariance matrix
        R:
          measurement noise covariance matrix
        P:
          error covariance matrix
        H:
          observation matrix
        """
        self.Q = get_diag_matrix(self.sigma_r, self.sigma_v, self.sigma_omega)
        self.R = get_diag_matrix(self.sigma_y)
        self.P = get_diag_matrix(self.ini_sigma_r, self.ini_sigma_v, self.ini_sigma_omega)
        
        self.H = np.hstack((np.identity(3), np.zeros((3, 6))))
        self.I = np.identity(9)

    def get_Jacobian(self, states: State9d, dt: float, kd: float, km: float) -> Matrix:
        """This function is used to generate the jacobian matrix of free falling motion

        Parameters
        ----------
        states:
          the states of the object
        dt:
          one single step
        kd:
          coefficient for the drag effect
        km:
          coefficient for the magnus effect
        """
        I = np.identity(3)
        Z = I * 0
        r = states[0:3]
        v = states[3:6]
        omega = states[6:9]

        v_norm = np.linalg.norm(v, ord=2)
        vx = v[0]
        vy = v[1]
        vz = v[2]

        omegax = omega[0]
        omegay = omega[1]
        omegaz = omega[2]

        F_1_drag = - dt*kd*np.array([[(v_norm + vx**2 / v_norm), (vy * vx / v_norm), (vz * vx / v_norm)],
                                     [(vy * vx / v_norm), (v_norm + vy**2 / v_norm), (vz * vy / v_norm)],
                                     [(vz * vx / v_norm), (vy * vz / v_norm), (v_norm + vz**2 / v_norm)]])
                        
        F_1_magnus = dt*km*np.array([[0.0, - omegaz, omegay],
                                     [omegaz, 0.0, - omegax],
                                     [- omegay, omegax, 0.0]])

        F_1 = I + F_1_drag + F_1_magnus

        F_2 = - dt * np.array([[v_norm * vx, 0.0, 0.0],
                            [0.0, v_norm * vy, 0.0],
                            [0.0, 0.0, v_norm * vz]])

        Jac = np.block([[I, dt*I,   Z],
                        [Z,  F_1, F_2],
                        [Z,    Z,   I]])
        return Jac
    
    @staticmethod
    def get_error_covariance(A: Matrix, P: Matrix, Q: Matrix) -> Matrix:
        return A@P@A.T + Q
    
    @staticmethod
    def get_output_difference(states_measurement: State9d, states_prediction: State9d, H: Matrix) -> State3d:
        return H@states_measurement - H@states_prediction
        
    @staticmethod
    def get_S(H: Matrix, P: Matrix, R: Matrix) -> Matrix:
        return np.linalg.inv(H@P@H.T + R)

    @staticmethod
    def get_kalman_gain(P: Matrix, H: Matrix, S: Matrix) -> Matrix:
        return P@H.T@S

    @staticmethod
    def get_next_states(states_prediction: State9d, K: Matrix, v: State3d) -> State9d:
        return states_prediction + K@v
    
    @staticmethod
    def update_error_covariance(I: Matrix, K: Matrix, H: Matrix, P: Matrix) -> Matrix:
        return (I - (K@H))@P
    
    def kalman_filter(self, states_prediction: State9d, 
                      states_measurement: State9d,
                      dt: float, kd: float, km: float) -> State9d:
        """This function is used to apply Kalman filter to estimate
        the states of the object at the next step

        Parameters
        ----------
        states_prediction:
          states of the object obtained from the model
        states_measurement:
          states of the object obtained from the sensors
        dt:
          one single time step
        kd:
          coefficient for drag effect
        km:
          coefficient for magnus effect
        
        Variables
        ---------
        A:
          Jacobian matrix obtained by linearizing the 
          model using measured states
        P:
          predicted error covariance matrix
        S:
          intermediate matrix used to calculate kalman gain
        K:
          kalman gain
        states:
          estimated states at the next time step
        
        Returns
        -------
        next states
        """
        A      = self.get_Jacobian(states_measurement, dt, kd, km)
        P      = self.get_error_covariance(A, self.P, self.Q)
        v      = self.get_output_difference(states_measurement, states_prediction, self.H)
        S      = self.get_S(self.H, P, self.R) 
        K      = self.get_kalman_gain(P, self.H, S)
        states = self.get_next_states(states_prediction, K, v)
        self.P = self.update_error_covariance(self.I, K, self.H, P)
        return states
    

    def state_estimation(self, free_falling: FreeFalling, 
                         states_last_step: State9d, 
                         states_measurement: State9d, dt: float) -> (State9d, tuple):
        """This function is used to estimate the state at the
        next step using Kalman filter, which includes impact 
        detection

        Parameters
        ----------
        free_falling:
          the class of free falling model
        states_last_step:
          the states of the object at the last step
        states_measurement:
          the current measured states of the object
        dt:
          one single time step

        Returns
        -------
        states_kf:
          estimated states at the next step
        states_before_impact:
          states of the object right before the impact
        states_after_impact:
          states of the object right after the impact
        impact_type:
          impact type
        """
        assert states_last_step.shape == (9,)
        assert states_measurement.shape == (9,)

        states_prediction, impact_info = free_falling.falling(states_last_step, dt)
        states_kf = self.kalman_filter(states_prediction, states_measurement, 
                                       dt, free_falling.kd, free_falling.km)
        return states_kf, impact_info
    
        
    
        
