""" Class and functions for states estimation
and motion prediction of a ball
"""

import numpy as np
import typing
import math
import json 

from kalman_filter import KalmanFilter
from free_falling import FreeFalling
import utils as fcs
from mytypes import Matrix, State3d, State6d, State9d, Array, Vector

def cal_section_area(radius: float) -> float:
    """ Given the radius of a ball, return the section area

    Parameters
    ----------
    radius:
      the radius of the ball
    
    Returns
    -------
    section_area:
      the section area of the ball
    """
    return math.pi*(radius**2)
 
class Ball:
    """ Instances of this object will be used to apply
    kalman filter to estimate the states of a ball,
    and to apply free falling model to prediction the
    rest motion of a ball within the prediction horizon

    Parameters
    ----------
    radius:
      the (common average) radius of the ball, in mm
    mass:
      the (common average) mass of the ball, in grams
    rho: 
      the (common average) density of the ball, in kg/m^3
    """

    def __init__(self) -> None:
        with open('config.json', 'r') as file:
            config = json.load(file)

        self.mass = config["ball"]["mass"]
        self.radius = config["ball"]["radius"]
        self.rho = config["ball"]["rho"]
        self.section_area = cal_section_area(self.radius)

        self.ball = {
            "t_stamp":            None,
            "states_measurement": None,
            "states_kf":          None,
            "states":             None,
            "states_impact":      None,
            "impact_type":        None
        }

        self.kalman_filter = KalmanFilter(config["kalman_filter"])
        self.free_falling = FreeFalling(config["free_falling"])
        self.reset()
    
    def reset(self) -> None:
        """ Reset the ball

        Parameters
        ----------
        t_ini:
          the initial time of a ball
        position_ini:
          the initial position of a ball
        empty_counter:
          counter used to record consecutive lost balls
        ball:
          a ball includes the measured position, velocity and spin,
          as well as the estimated position, velocity and spin
        """
        self.t_ini = None
        self.position_ini = None
        self.empty_counter = 0
        for key in self.ball.keys():
            self.ball[key] = None

        self.kalman_filter.reset()
    
    def ball_initialization(self, t: float, position: State3d, velocity: State3d, omega: State3d) -> None:
        """ Initialize a ball

        Parameters
        ----------
        t:
          considered as the initial time of a ball
        position:
          considered as the initial position of a ball
        velocity:
          considered as the initial velocity of a ball
        """
        assert position.shape == (3,)
        assert velocity.shape == (3,)
        assert omega.shape == (3,)
        assert self.t_ini is None
        assert self.position_ini is None
        for key in self.ball.keys():
            assert self.ball[key] is None

        self.t_ini = t
        self.position_ini = position
        states = fcs.merge_arrays(position, velocity, omega)
        self.ball = {
            "t_stamp":            [t],
            "states_measurement": [states],
            "states_kf":          [states],
            "states":             states,
            "states_impact":      [],
            "impact_type":        []
        }

    def continuity_judgment(self, t: float, position: State3d, velocity: State3d, omega: State3d) -> bool:
        """ This function is used to determine whether the ball is detected. 
        If the ball is not detected, then determine whether the ball has been 
        lost too many times (empty_counter > 20). 
        If a ball is detected, then determine whether it is a new ball 
        (displacement is too large).

        Parameters
        ----------
        t:
          current time stamp
        position:
          current measured position of the ball
        velocity:
          current measured velocity of the ball
        omega:
          current measured spin of the ball
        
        Returns
        -------
        bool
        """
        if t == -1 and self.t_ini is None:  # no measurement and no previous ball exists
            self.empty_counter = 0
            return 0
        elif t == -1 and self.t_ini is not None:  # no measurement and previous ball exists
            self.empty_counter += 1
            if self.empty_counter > 20:  # lose the ball in 20 consecutive steps
                self.reset()
            return 0
        elif t != -1 and self.t_ini is None:  # measurement exists and no previous ball exists
            self.empty_counter = 0
            self.ball_initialization(t, position, velocity, omega)
            return 0
        elif t != -1 and self.t_ini is not None:  # measurement exists and previous ball exists
            if np.linalg.norm(position - self.ball["states_kf"][-1][0:3]) > 0.5:  # it is too far away from the previous position and is judged as a new ball
                self.reset()
                self.ball_initialization(t, position, velocity)
                return 0
            else:
                self.empty_counter = 0
                return 1

    def extend_array(self, array, element):
        return np.concatenate((array, element.reshape(1, -1)), axis=0)
    
    def ball_update(self, t: float, states_kf: State9d, states_measurement: State9d, 
                    states_before_impact: State9d, 
                    states_after_impact: State9d, 
                    impact_type: str) -> None:
        """ This function is used to update the states of the ball

        Parameters
        ----------
        t:
          current time stamp
        states_kf:
          the states of the ball from kalman filter
        states_measurement:
          the states of the ball from sensors
        states_before_impact, states_after_impact:

        """
        self.ball["t_stamp"].append(t)
        self.ball["states"] = states_kf
        self.ball["states_measurement"].append(states_measurement)
        self.ball["states_kf"].append(states_kf)
        if impact_type is not None:
            self.ball["impact"].append(impact_type)
            self.ball["states_impact"].append((states_before_impact, states_after_impact))
    
    def input_data(self, t: float, position_measurement: State3d, velocity_measurement: State3d, omega_measurement: State3d) -> None:
        """ Perform a reading of the state of the ball from the visual system, 
        first judge the continuity of the ball, and then use Kalman filter to 
        estimate the state of the ball.

        Parameters
        ----------
        t:
          current time stamp
        position_measurement:
          current measured position of the ball
        velocity_measurement:
          current measured velocity of the ball
        omega_measurement:
          current measured spin of the ball
        """
        assert position_measurement.shape == (3,)
        assert velocity_measurement.shape == (3,)
        assert omega_measurement.shape == (3,)

        _ball_continuity = self.continuity_judgment(t, position_measurement, velocity_measurement, omega_measurement)
        if _ball_continuity == 1:
            dt = t - self.ball["t_stamp"][-1]
            states_measurement = fcs.merge_arrays(position_measurement, velocity_measurement, omega_measurement)
            states_kf, states_before_impact, states_after_impact, impact_type = self.kalman_filter.state_estimation(self.free_falling, self.ball["states"], states_measurement, dt)
            self.ball_update(t, states_kf, states_measurement, states_before_impact, states_after_impact, impact_type)
    
    @staticmethod
    def _prediction(free_falling: FreeFalling, t_ini: float, states: State9d, prediction_horizon: float) -> Array:
        pass

    def prediction(self, prediction_horizon: float) -> Array:
        pass



        

        
      
    
