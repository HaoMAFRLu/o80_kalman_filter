"""Classes and functions for free fall motion 
and detecting the impact with the environment
"""

import numpy as np

from environment import Table, Ground
from mytypes import Matrix, State3d, State6d, State9d, Array


class FreeFalling():
    """ Instances of this object will be used to apply
    the model of free falling and detect the impact with
    the environment
    and to apply free falling model to prediction the
    rest motion of a ball within the prediction horizon

    Parameters
    ----------
    gravity:
      the gravity along negative z-axis, in m/s^2
    kd:
      coefficient for drag effect
    km:
      coefficient for magnus effect
    """
    def __init__(self, config) -> None:
        self.gravity = config["gravity"]
        self.kd = config["kd"]
        self.km = config["km"]

        self.table = Table(config["table"])
        self.ground = Ground(config["ground"])
    
    def time_for_motion(self, vini: float, a: float, dis: float) -> float:
        """This function is used to calculate the time required 
        to reach a specified displacement under a given initial 
        velocity and acceleration

        Parameters
        ----------
        vini:
          initial velocity, m/s
        a:
          acceleration, m/s^2
        dis
          displacement, m

        Returns
        -------
        required time
        """
        if vini**2 + 2*a*dis < 0:
            t =  0.0
        else:
            t = (-vini - np.sqrt(vini**2 + 2*a*dis))/a
        assert t >= 0.0
        return t

    def impact_detection(self, states: State9d) -> (float, str):
        """This function is used to determine the time required 
        for the object to impact with the environment in its 
        current state.
        
        Parameters
        ----------
        states:
          the states of the object
        
        Returns
        -------
        t:
          time required until the impact happens
        impact_type:
          impact type
        """
        if self.table.region_check(states):
            height = self.table.center[2]
            impact_type = 'table'
        else:
            height = self.ground.height
            impact_type = 'ground'

        vz = states[5]
        height_difference = height - states[2]
        t = self.time_for_motion(vz, -self.gravity, height_difference)
        return t, impact_type
    
    def free_falling(self, states: State9d, dt: float) -> State9d:
        """This function is used to apply free falling motion
        without any impact

        Parameters
        ----------
        states:
          the states of the object
        dt:
          one single step
        
        Returns
        -------
        Next states
        """
        v_norm = np.linalg.norm(states[3:6], ord=2)
        states_next = np.hstack((states[0] + states[3]*dt, 
                                 states[1] + states[4]*dt, 
                                 states[2] + states[5]*dt - 0.5*self.gravity*dt**2, 
                                 states[3] + dt*(-self.kd*v_norm*states[3] + self.km*(states[7]*states[5] - states[8]*states[4])) , 
                                 states[4] + dt*(-self.kd*v_norm*states[4] + self.km*(states[8]*states[3] - states[6]*states[5])) , 
                                 states[5] + dt*(-self.kd*v_norm*states[5] + self.km*(states[6]*states[4] - states[7]*states[3]) - self.gravity),  
                                 states[6],
                                 states[7],
                                 states[8],))
        return states_next

    def cal_impact(self, states: State9d, impact_type: str) -> State9d:
        """This function is used to calculate the state 
        of the object after the impact based on the 
        given collision type.
        
        Parameters
        ----------
        states:
          the states of the object right before the impact
        impact_type
          the impact type
        
        Returns
        -------
        states_next:
          the states of the object right after the impact
        """
        if impact_type == 'table':
            states_next = self.table.impact(states)
        elif impact_type == 'ground':
            states_next = self.ground.impact(states)
        return states_next

    def falling(self, states: State9d, dt: float) -> (State9d, tuple):
        """This function is used to apply the free 
        fall motion for one single time step dt and 
        detect the impact with the environment. 
        The function returns the states of the object 
        before and after the impact, as well as
        the impact type.

        Parameters
        ----------
        states:
          states of the object from last time step
        dt:
          one single time step

        Returns
        -------
        states_falling:
          the state of the object after one time step 
          according to free falling motion
        states_before_impact:
          the states of the object right before the impact
        states_after_impact:
          the states of the object right after the impact
        impact_type:
          type of the impact
        """
        assert states.shape == (9,)

        t_until_impact, impact_type = self.impact_detection(states)
        if t_until_impact >= dt: # no impact will happen in one single step
            states_falling = self.free_falling(states, dt)
            states_before_impact = None
            states_after_impact = None
            impact_type = None
        else:
            states_before_impact = self.free_falling(states, t_until_impact)
            states_after_impact = self.cal_impact(states_before_impact, impact_type)
            states_falling = self.free_falling(states_after_impact, dt-t_until_impact)
        
        impact_info = (states_before_impact, states_after_impact, impact_type)
        return states_falling, impact_info
