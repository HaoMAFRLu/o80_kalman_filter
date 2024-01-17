import math
import time
import numpy as np
import json

from environment import Table, Ground

def get_diag_matrix(lists_tuple):
    """
    Create a diagonal matrix from a tuple of lists.

    :param lists_tuple: (tuple) A tuple containing lists. Each list represents diagonal elements.
    :return: (numpy.ndarray) A diagonal matrix.
    """
    # Determine the size of the matrix
    size = sum(len(lst) for lst in lists_tuple)
    # Create an empty matrix of the determined size
    diag_matrix = np.zeros((size, size))
    # Fill the diagonal elements
    start_index = 0
    for lst in lists_tuple:
        length = len(lst)
        end_index = start_index + length
        np.fill_diagonal(diag_matrix[start_index:end_index, start_index:end_index], lst)
        start_index += length
    return diag_matrix


class KalmanFilter:
    def __init__(self) -> None:
        with open('config.json', 'r') as file:
            config = json.load(file)
        
        self.gravity = config["gravity"]
        self.frequency = config["frequency"]
        self.prediction_horizon = config["prediction_horizon"]
        self.dt = 1/self.frequency
        self.kd = config["kd"]
        self.km = config["km"]

        self.ini_omega = config["ini_omega"]
        
        self.sigma_r = config["sigma"]["r"]
        self.sigma_v = config["sigma"]["v"]
        self.sigma_y = config["sigma"]["y"]
        self.sigma_omega = config["sigma"]["omega"]

        self.ini_sigma_r = config["ini_sigma"]["r"]
        self.ini_sigma_v = config["ini_sigma"]["v"]
        self.ini_sigma_omega = config["ini_sigma"]["omega"]

        self.ball = Ball(config["ball"])
        self.table = Table(config["table"])
        self.ground = Ground(config["table"])

    

    def reset_kalman_filter(self, prediction_horizon=None):
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        num_points = prediction_horizon*self.frequency
        self.position = np.zeros((3, num_points))
        self.position_kf = np.zeros((3, num_points))
        self.velocity_kf = np.zeros((3, num_points))
        self.omega_kf = np.zeros((3, num_points))

        self.time_past = int(0)
        self.states_meas = None
        self.r_meas = None
        self.x_pred = None

        self.Q = get_diag_matrix(self, (self.sigma_r, self.sigma_v, self.sigma_omega))
        self.R = get_diag_matrix(self, (c))
        self.P_m = get_diag_matrix(self, (self.ini_sigma_r, self.ini_sigma_v, self.ini_sigma_omega))
        
        self.H = np.hstack((np.identity(3), np.zeros((3, 6))))
        self.I = np.identity(9)

        self.t_to_impact = 100  # big enough
        self.impact_flag = False
        self.t_stamp = None
        self.impact_time_list = []
        self.impact_type = []
        self.state_before_impact = []
        self.state_after_impact = []

    def GetInitial(self, r, v):
        self.reset_kalman_filter()
        self.position[:, 0] = r
        self.position_kf[:, 0] = r
        self.velocity_kf[:, 0] = v
        self.omega_kf[:, 0] = self.ini_omega

    def Falling(self, x, dt=None):
        if dt is None:
            dt = self.dt
        
        v_norm = np.linalg.norm(x[3:6], ord=2)
        x_est = np.hstack((x[0] + x[3] * dt, 
                           x[1] + x[4] * dt, 
                           x[2] + x[5] * dt - 0.5 * self.gravity * dt**2, 
                           x[3] + dt * (-self.kd * v_norm * x[3] + self.km * (x[7]*x[5] - x[8]*x[4])) , 
                           x[4] + dt * (-self.kd * v_norm * x[4] + self.km * (x[8]*x[3] - x[6]*x[5])) , 
                           x[5] + dt * (-self.kd * v_norm * x[5] + self.km * (x[6]*x[4] - x[7]*x[3]) - self.gravity),  
                           x[6],
                           x[7],
                           x[8],)).reshape(-1, 1)

        return x_est

    def JacobianMatrix(self, x, dt=None):
        if dt is None:
            dt = self.dt

        # define following for easier notation
        I = np.identity(3)
        Z = I * 0

        v_norm = np.linalg.norm(x[3:6], ord=2)

        vx = x[3].item()
        vy = x[4].item()
        vz = x[5].item()

        omegax = x[6].item()
        omegay = x[7].item()
        omegaz = x[8].item()

        F_1_drag = - dt * self.kd * np.array([[(v_norm + vx**2 / v_norm), (vy * vx / v_norm), (vz * vx / v_norm)],
                                             [(vy * vx / v_norm), (v_norm + vy**2 / v_norm), (vz * vy / v_norm)],
                                             [(vz * vx / v_norm), (vy * vz / v_norm), (v_norm + vz**2 / v_norm)]])
                        
        F_1_magnus = dt * self.km * np.array([[0.0, - omegaz, omegay],
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

    def ImpactDetection(self, x=None, dt=None):
        if x is None:
            x = np.copy(self.x_meas)
        if T is None:
            T = self.frequency
        if self.table.region_check(x):
            height_limit = self.table.center[2]
        else:
            height_limit = self.table.ground
        # velocity and position of the ball in z-axis
        vz = x[5]
        rz = x[2]
        height_difference = height_limit - rz
        # introduce nonlinear model???
        if vz**2 - 2*self.gravity*height_difference < 0:
            return 0.0
        t = (vz + np.sqrt(vz**2 - 2*self.gravity*height_difference)) / self.gravity
        return t  


    def single_prediction(self, states_before, dt):
        states_after = self.Falling(states_before, dt)

        if self.table.region_check(states_after):
            height_limit = self.table.center[2]
            type = 'table'
        else:
            height_limit = self.table.ground
            type = 'ground'

        if states_after[2] < height_limit:
            rz = states_before[2]
            vz = states_before[5]
            height_difference = height_limit - rz
            temp = vz**2 - 2*self.gravity*height_difference
            
            if temp < 0:
                t_to_impact = 0
            else:
                t_to_impact = (vz + np.sqrt(vz**2 - 2*self.gravity*height_difference))/self.gravity

            states_after = self.Falling(states_before, t_to_impact)
            if type == 'table':
                states_after = self.table.impact_table(states_after)
            elif type == 'ground':
                states_after = self.table.impact_ground(states_after)
            states_after = self.Falling(states_after, dt-t_to_impact)
        else:
            type = None
        
        return states_after, type
        
    def multi_prediction(self, states=None, prediction_horizon=3, time_past=None):
        '''
        states:  the start states of the ball
        prediction_horizon: how long time to predict, in s
        time_past: how long time has past, in s
        '''
        if states is None:
            states = np.copy(self.states_meas)
        if time_past is None:
            time_past = self.time_past

        trajectory = states[0:3].rehspe(-1, 1)
        t_stamp = np.array([time_past])
        impact_time_list = []
        impact_type = []

        for i in range(len(self.impact_time_list)):
            if self.impact_time_list[i] < time_past:
                impact_time_list.append(np.float(self.impact_time_list[i]))
                impact_type.append(self.impact_type[i])
            else:
                break

        if prediction_horizon < time_past:
            print(prediction_horizon, time_past)
            print('No more prediction!')
            return (trajectory, t_stamp, impact_time_list, impact_type)  
        else:
            self.states_kf_before_impact = None
            states_before = np.copy(states)
            step_left = (prediction_horizon-time_past)*self.frequency
            
            for step in range(step_left):
                time_past += self.dt
                states_after, type = self.single_prediction(states_before, self.dt)

                if type is not None:
                    impact_time_list.append(np.float(time_past))
                    impact_type.append(type)

                t_stamp = np.append(t_stamp, time_past)
                trajectory = np.concatenate((trajectory, states_after[0:3].reshape(-1, 1)), axis=1)
                states_before = np.copy(states_after)

            return (trajectory, t_stamp, impact_time_list, impact_type)    

    def StateEstimation(self, r, v, k=None, dt=None):
        # r for positions from sensor, v for velocities from sensor
        if dt is None:
            dt = self.dt

        if self.time_past is None:
            self.GetInitial(r, v)
            self.states_meas = np.hstack((self.position_kf[:, 0],
                                     self.velocity_kf[:, 0],      
                                     self.omega_kf[:, 0])).reshape(-1, 1)
            self.t_stamp = np.array([self.time_past])
            self.t_to_impact = self.ImpactDetection()  # detect the impact time for the next step

        else:
            if self.t_to_impact > dt:
                self.impact_flag = False
            else:
                self.impact_flag = True

            # Kalman Filter
            self.r_meas = r.reshape(-1, 1)
            
            if self.impact_flag == False:
                self.states_pred = self.Falling(self.states_meas, dt)
              
            else: # perform an impact 
                self.impact_time_list.append(k)
                self.states_pred = self.Falling(self.states_meas, self.t_to_impact)

                self.states_before_impact.append(self.states_meas[0:15])

                if self.table.region_check(states=self.states_pred):
                    self.states_pred = self.table.impact_table(self.states_pred).reshape(-1, 1)
                    self.impact_type.append('table')
                else:
                    self.x_pred = self.table.impact_ground(self.states_pred)
                    self.impact_type.append('ground')

                self.states_pred = self.Falling(self.states_pred, dt-self.t_to_impact)

            if len(self.impact_time_list) > 0:
                if k - self.impact_time_list[-1] < 2:
                    Q = 50 * self.Q
                else:
                    Q = self.Q
            else:
                Q = 1*self.Q

            A = self.JacobianMatrix(self.states_meas, dt)
            P_p = A@self.P_m@A.T + Q
            K = P_p@self.H.T @ np.linalg.inv(self.H@P_p@self.H.T + self.R)
            states_before = self.states_meas

            self.states_meas = np.array(self.states_pred + K@(self.r_meas - self.H@self.states_pred)).reshape(-1, 1)
            self.P_m = (self.I - (K@self.H))@P_p

            self.time_past += dt
            
            if len(self.t_stamp) >= self.position_kf.shape[1]:
                return

            self.position_kf[:, len(self.t_stamp)] = self.states_meas[0:3].flatten()
            self.velocity_kf[:, len(self.t_stamp)] = self.states_meas[3:6].flatten()
            self.omega_kf[:, len(self.t_stamp)]    = self.states_meas[6:9].flatten()

            self.t_stamp = np.append( self.t_stamp, self.time_past )
            # predict the impact time for the next step
            self.t_to_impact = self.ImpactDetection()
            if self.impact_flag == True:
                self.impact_flag = False
                self.state_after_impact.append( self.x_meas[0:9] )
    
    # def Plot2DEstimation( self ):
    
    #     # print(self.time_past)
    #     # print(self.t_stamp)
    #     # print(self.position_kf)
    #     speed_list = [np.linalg.norm(self.velocity_kf[:, i]) for i in range(0, 150)]

    #     fig = plt.figure( figsize=(16, 8) )
    #     plt.plot(speed_list)
    #     plt.show()


    #     fig = plt.figure( figsize=(16, 8) )
    #     # plt.plot(self.t_stamp, self.position_kf[0, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated x')
    #     # plt.plot(self.t_stamp, self.position_kf[:,0] + self.var_position[:, 0] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position_kf[:,0] - self.var_position[:, 0] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position[0, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'x')

    #     # plt.plot(self.t_stamp, self.position_kf[1, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated y')
    #     # plt.plot(self.t_stamp, self.position_kf[:,1] + self.var_position[:, 1] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position_kf[:,1] - self.var_position[:, 1] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position[1, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'y')

    #     # plt.plot(self.t_stamp, self.position_kf[2, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated z')
    #     # plt.plot(self.t_stamp, self.position_kf[:,2] + self.var_position[:, 2] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position_kf[:,2] - self.var_position[:, 2] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.position[2, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'z')
        
    #     #plt.xlabel('Time in t')
    #     #plt.ylabel('Position in m')
    #     #plt.legend(ncol=7, loc='upper center')
    #     #plt.show()

    #     # fig = plt.figure( figsize=(16, 8) )
    #     # plt.plot(self.t_stamp, self.velocity_kf[0, :] , linewidth=0.5, label=r'estimated x velocity')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,0] + self.var_velocity[:, 0] , linewidth=0.5, linestyle='--')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,0] - self.var_velocity[:, 0] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.velocity[0, :] , linewidth=1, linestyle='--', label=r'x velocity')

    #     # plt.plot(self.t_stamp, self.velocity_kf[1 ,:] , linewidth=0.5, label=r'estimated y velocity')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,1] + self.var_velocity[:, 1] , linewidth=0.5, linestyle='--')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,1] - self.var_velocity[:, 1] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.velocity[1, :] , linewidth=1, linestyle='--', label=r'y velocity')

    #     # plt.plot(self.t_stamp, self.velocity_kf[2, :] , linewidth=0.5, label=r'estimated z velocity')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,2] + self.var_velocity[:, 2] , linewidth=0.5, linestyle='--')
    #     # # plt.plot(self.t_stamp, self.velocity_kf[:,2] - self.var_velocity[:, 2] , linewidth=0.5, linestyle='--')
    #     # plt.plot(self.t_stamp, self.velocity[2, :] , linewidth=1, linestyle='--', label=r'z velocity')
      
    #     # plt.xlabel('Time in t')
    #     # plt.ylabel('Velocity in m/s')
    #     # plt.legend(ncol=7, loc='upper center')
    #     # plt.show()

    # def _set_axes_radius(self, ax, origin, radius):
    #     x, y, z = origin
    #     ax.set_xlim3d([x - radius, x + radius])
    #     ax.set_ylim3d([y - radius, y + radius])
    #     ax.set_zlim3d([z - radius, z + radius])
        
    # def set_axes_equal(self, ax: plt.Axes):
    #     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    #     origin = np.mean( limits, axis=1 )
    #     radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    #     self._set_axes_radius(ax, origin, radius)

    # def Plot3DEstimation( self, y_sensor=None, y_to_compare=None, y_real=None, if_table='yes' ):

    #     if y_real is None:
    #         y_real = np.copy(self.position_kf)
    #     if y_to_compare is None:
    #         y_to_compare = [np.copy(self.position_kf)]

    #     fig = plt.figure( figsize=(8, 8) )
    #     ax = plt.subplot(111, projection='3d')

    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['top'].set_linewidth(1.5)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['right'].set_linewidth(1.5)

    #     for i in range(len(y_to_compare)):
    #         if i%10 == 0 or True:
    #             ax.scatter(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :], c='red', s=0.5)
    #             ax.plot3D(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :],'grey', linewidth=0.1)
    #             #ax.scatter(y_to_compare[0, :], y_to_compare[1, :], y_to_compare[2, :], c='red', s=0.5)
    #             #ax.plot3D(y_to_compare[0, :], y_to_compare[1, :], y_to_compare[2, :],'grey', linewidth=0.1)

    #     #ax.scatter(y_sensor[0, 0:len(self.t_stamp)], y_sensor[1, 0:len(self.t_stamp)], y_sensor[2, 0:len(self.t_stamp)], c='blue', s=5)
    #     #ax.plot3D(y_sensor[0, 0:len(self.t_stamp)], y_sensor[1, 0:len(self.t_stamp)], y_sensor[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'trajectory')
    #     #ax.scatter(y_sensor[0, 0], y_sensor[1, 0], y_sensor[2, 0], c='g', s=50 )

    #     ax.scatter(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)], c='black', s=5)
    #     ax.plot3D(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'real')

    #     plt.legend(ncol=1, loc='upper center', shadow=True, fontsize=14)
    #     ax.set_xlabel(r'$x$ in m', fontsize=14)
    #     ax.set_ylabel(r'$y$ in m', fontsize=14)   
    #     ax.set_zlabel(r'$z$ in m', fontsize=14)
    #     self.set_axes_equal(ax)   

    #     # draw the table
    #     if if_table == 'yes':
    #         v1 = self.center_of_table + self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
    #         v1 = np.append(v1, self.height_of_ground + self.height_of_table)

    #         v2 = self.center_of_table + self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
    #         v2 = np.append(v2, self.height_of_ground + self.height_of_table)

    #         v3 = self.center_of_table - self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
    #         v3 = np.append(v3, self.height_of_ground + self.height_of_table)

    #         v4 = self.center_of_table - self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
    #         v4 = np.append(v4, self.height_of_ground + self.height_of_table)

    #         vtx = list([tuple(v1), tuple(v3), tuple(v4) , tuple(v2)])
    #         table_area = Poly3DCollection([vtx], facecolors='lightsteelblue', linewidths=1, alpha=0.5) 
    #         # table_area.set_color('linen')
    #         table_area.set_edgecolor('k')
    #         ax.add_collection3d( table_area )

    #     #plt.show()

    # def PlotHitPoint(self, y=None, hit_position=None, if_table='yes', if_traj='yes'):
        if y is None:
            y = np.copy(self.position_kf)
        
        fig = plt.figure( figsize=(8, 8) )
        ax = plt.subplot(111, projection='3d')

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        ax.scatter(hit_position[0, :], hit_position[1, :], hit_position[2, :],c='r', s=20)
        
        if if_traj=='yes':
            ax.scatter(y[0, 0:len(self.t_stamp)], y[1, 0:len(self.t_stamp)], y[2, 0:len(self.t_stamp)], c='blue', s=5)
            ax.plot3D(y[0, 0:len(self.t_stamp)], y[1, 0:len(self.t_stamp)], y[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'trajectory')
            ax.scatter(y[0, 0], y[1, 0], y[2, 0], c='g', s=50 )

        plt.legend(ncol=1, loc='upper center', fontsize=14)
        ax.set_xlabel(r'$x$ in m', fontsize=14)
        ax.set_ylabel(r'$y$ in m', fontsize=14)   
        ax.set_zlabel(r'$z$ in m', fontsize=14)
        self.set_axes_equal(ax)   

        # draw the table
        if if_table == 'yes':
            v1 = self.center_of_table + self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v1 = np.append(v1, self.height_of_ground + self.height_of_table)

            v2 = self.center_of_table + self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v2 = np.append(v2, self.height_of_ground + self.height_of_table)

            v3 = self.center_of_table - self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v3 = np.append(v3, self.height_of_ground + self.height_of_table)

            v4 = self.center_of_table - self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v4 = np.append(v4, self.height_of_ground + self.height_of_table)

            vtx = list([tuple(v1), tuple(v3), tuple(v4) , tuple(v2)])
            table_area = Poly3DCollection([vtx], facecolors='lightsteelblue', linewidths=1, alpha=0.5) 
            # table_area.set_color('linen')
            table_area.set_edgecolor('k')
            ax.add_collection3d( table_area )

        plt.show()
        
