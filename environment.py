""" Classes of objects in the lab, e.g. table, racket, for detecting
    the contact with the ball.
"""

import numpy as np
import typing
import math
import json 

from mytypes import Matrix, State3d, State6d, State9d

def cal_normalized_direction(direction: State3d) -> State3d:
    return direction/np.linalg.norm(direction, ord=2)

class Table():
    """ Instances of this object will be used as a table
    for detecting the contact

    Parameters
    ----------
    center:
      center of the table in the vision system
    height:
      height of the table, in m
    direction:
      direction of the long side of the table
    half_length:
      half length of the table, in m
    half_width:
      half width of the table, in m
    """
    def __init__(self, config) -> None: 
        self.center = config["center"]
        self.height = config["height"]
        self.direction = config["direction"]
        self.half_length = config["half_length"]
        self.half_width = config["half_width"]
        self.model = config["impact_table"]
        self.normalized_direction = cal_normalized_direction(self.direction)

    def cal_impact(self, states: State9d, model) -> State9d:
        """ Applies the linear impact model on the given states
        
        Parameters
        ----------
        states:
          9d state (3d position, 3d velocity and 3d omega)

        Returns
        -------
        Next state
        """
        states[3:6] = model@states[3:6].reshape(-1, 1)
        return states
        

    def impact(self, states: State9d, model=None) -> State9d:
        """ Applies the impact model on the provided state 
        in the coordinate system of vision
        
        Parameters
        ----------
        states:
          9d state (3d position, 3d velocity and 3d omega)
        model:
          impact model, linear or non-linear

        Returns
        -------
        Next states
        """
        assert states.shape == (9,)

        if model is None:  # no specified model
            model = self.model

        return self.cal_impact(states, model)
    
    def is_on_table(self, point2d, length, width) -> bool:
        """ 
        Parameters
        ----------
        point2d:
          the projection of a 3d point in the table plane

        Returns
        -------
        bool
        """
        return -length <= point2d[0] <= length and -width <= point2d[1] <= width

    def region_check(self, states: State9d) -> bool:
        """ Check if the projection of the point remains in the table
        
        Parameters
        ----------
        point:
          states

        Returns
        -------
        bool
        """
        assert states.shape == (9,)

        position = states[0:3]
        vector = (position[0] - self.center[0], position[1] - self.center[1])
        dot_length = vector[0]*self.normalized_direction[0] + vector[1]*self.normalized_direction[1]
        dot_width = vector[0]*self.normalized_direction[1] - vector[1]*self.normalized_direction[0]
        return self.is_on_table((dot_length, dot_width), self.half_length, self.half_width)

class Ground():
    """ Instances of this object will be used as the ground
    for detecting the contact

    Parameters
    ----------
    height:
      height of the table, in m
    direction:
      normal direction
    """
    def __init__(self, config) -> None:
        center = config["center"]
        height = config["height"]
        self.height = center[2] - height
        self.model = config["impact_ground"]
    
    def cal_impact(self, states: State9d, model) -> State9d:
        """ Applies the linear impact model on the given states
        
        Parameters
        ----------
        states:
          9d state (3d position, 3d velocity and 3d omega)

        Returns
        -------
        Next state
        """
        states[3:6] = model@states[3:6].reshape(-1, 1)
        return states
        

    def impact(self, states: State9d, model=None) -> State9d:
        """ Applies the impact model on the provided state 
        in the coordinate system of vision
        
        Parameters
        ----------
        states:
          9d state (3d position, 3d velocity and 3d omega)

        Returns
        -------
        Next states
        """
        assert states.shape == (9,)

        if model is None:  # no specified model
            model = self.model

        return self.cal_impact(states, model) 

class Racket():
    def __init__(self) -> None:
        pass
