'''
Author: Qin Yang
05/08/2021
'''

#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
from fractions import Fraction
import time
import os
import random
import greedy_single3

def normal_situation(N, L, situation):
    if situation == 'patroling':
        agents = -lineGL(N)
        L[0:N,0:N] = agents

    return L

def greedy_strategy_selector(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level):
    strategy = []

    tmp = sorted(greedy_single3.Greedy_DecisionMaking(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level).items(), key = lambda kv:(kv[1], kv[0]))[-1]
    print(tmp[0])


    if tmp[0] == 'ad':
        strategy.append('attacking')
        strategy.append('change_direction')
    elif tmp[0] == 'as':
        strategy.append('attacking')
        strategy.append('change_speed')
    elif tmp[0] == 'dd':
        strategy.append('defending')
        strategy.append('change_direction')
    elif tmp[0] == 'ds':
        strategy.append('defending')
        strategy.append('change_speed')

    return strategy  

def alien_detector(numExplorer, x, id, sensing_distance):
    detector = False

    for i in range(numExplorer):
        if np.linalg.norm(x[:2,[i]] - x[:2,[id]]) < sensing_distance:
            detector = True

    return detector

def gut_explore_game():
    # Experiment Constants
    iterations = 5000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
    N=7 #Number of robots to use, this must stay 4 unless the Laplacian is changed.

    waypoints = np.array([[-1.058, -0.977, 1.37],[-0.53, 0.6, 0.8]]) #Waypoints the leader moves to.

    alienpoints1 = np.array([[-1.398, -1.298],[-0.809, -0.362]]) # attacking
    alienpoints2 = np.array([[-1.398, -1.198],[-0.362, -0.043]]) # attacking
    alienpoints3 = np.array([[-0.662, -1.052],[0.876, 0.8]]) # defending

    obstaclepoints = np.array([[0.936, -0.025],[-0.06, -0.166]])

    close_enough = 0.03; #How close the leader must get to the waypoint to move to the next one.

    # sensing distance between explorer and alien
    sensing_distance = 0.7

    #Initialize leader state
    state = 0
    state1 = 0
    stateList = [0] * 4
    alienState = 0
    numActiveAlien = 0
    activeAlien = []

    unit_explorer_attacking_hp_cost = 0.03
    unit_alien_attacking_hp_cost = 0.05

    # initial agent's energy and hp
    agent_energy_level = []
    agent_hp_level = []


    for i in range(N):
        if i < N-3:
            agent_energy_level.append(100)
            agent_hp_level.append(100)
        else:
            agent_energy_level.append(150)
            agent_hp_level.append(150)
        

    #Limit maximum linear speed of any robot
    magnitude_limit = 0.15

    # Create gains for our formation control algorithm
    formation_control_gain = 10
    desired_distance = 0.3

    # Initial Conditions to Avoid Barrier Use in the Beginning.
    initial_conditions = np.array([[0.55, 0.85, 1.15, 1.45, -1.283, -1.398, -1.368],[-0.85, -0.85, -0.85, -0.85, 0.817, -0.809, -0.362],[0, 0, 0, 0, 0, 0, 0]])

    # Instantiate the Robotarium object with these parameters
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

    # Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
    # Single-integrator -> unicycle dynamics mapping
    _,uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    # Single-integrator barrier certificates
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    # Single-integrator position controller
    explorer_controller = create_si_position_controller(velocity_magnitude_limit=0.1)
    alien1_controller = create_si_position_controller(velocity_magnitude_limit=0.3)
    alien2_controller = create_si_position_controller(velocity_magnitude_limit=0.3)
    alien3_controller = create_si_position_controller(velocity_magnitude_limit=0.3)

    # For computational/memory reasons, initialize the velocity vector
    si_velocities = np.zeros((2, N))

    # Plotting Parameters
    # Random Colors
    CM1 = np.random.rand(N,3)
    CM2 = np.random.rand(N,3)
    CM3 = np.random.rand(N,3)
    marker_size_goal = determine_marker_size(r,0.2)
    robot_marker_size_m = 0.15
    font_size_m = 0.1
    font_size = determine_font_size(r,font_size_m)
    font_size_m1 = 0.06
    font_size1 = determine_font_size(r,font_size_m1)
    font_size_m2 = 0.04
    font_size2 = determine_font_size(r,font_size_m2)
    marker_size_robot = determine_marker_size(r, robot_marker_size_m)
    line_width = 5

    # Create goal text and markers

    #Text with goal identification
    goal_caption = ['Encounter', 'Encounter', 'Treasure']
    obstacle_caption = ['Obstacle', 'Obstacle', 'Obstacle']
    #Plot text for caption
    waypoint_text = [r.axes.text(waypoints[0,ii], waypoints[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
    for ii in range(waypoints.shape[1])]
    g = [r.axes.scatter(waypoints[0,ii], waypoints[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM1[ii,:],linewidth=line_width,zorder=-2)
    for ii in range(waypoints.shape[1])]

    obstaclepoint_text = [r.axes.text(obstaclepoints[0,ii], obstaclepoints[1,ii], obstacle_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
    for ii in range(obstaclepoints.shape[1])]
    g1 = [r.axes.scatter(obstaclepoints[0,ii], obstaclepoints[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM3[ii,:],linewidth=line_width,zorder=-2)
    for ii in range(obstaclepoints.shape[1])]

    # Plot Graph Connections
    x = r.get_poses() # Need robot positions to do this.
    old_x = []

    for i in range(N):
        old_x.append(initial_conditions[:2, [i]])


    explorer1_label = r.axes.text(x[0,0],x[1,0]+0.25,"explorer 1",fontsize=font_size1, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer1_energy_label = r.axes.text(x[0,0],x[1,0]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer1_hp_label = r.axes.text(x[0,0],x[1,0]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    explorer2_label = r.axes.text(x[0,1],x[1,1]+0.25,"explorer 2",fontsize=font_size1, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer2_energy_label = r.axes.text(x[0,1],x[1,1]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer2_hp_label = r.axes.text(x[0,1],x[1,1]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    explorer3_label = r.axes.text(x[0,2],x[1,2]+0.25,"explorer 3",fontsize=font_size1, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer3_energy_label = r.axes.text(x[0,2],x[1,2]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer3_hp_label = r.axes.text(x[0,2],x[1,2]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    explorer4_label = r.axes.text(x[0,3],x[1,3]+0.25,"explorer 4",fontsize=font_size1, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer4_energy_label = r.axes.text(x[0,3],x[1,3]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer4_hp_label = r.axes.text(x[0,3],x[1,3]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    alien1_label = r.axes.text(x[0,4],x[1,4]+0.25,"alien 1",fontsize=font_size1, color='y',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien1_energy_label = r.axes.text(x[0,4],x[1,4]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien1_hp_label = r.axes.text(x[0,4],x[1,4]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    alien2_label = r.axes.text(x[0,5],x[1,5]+0.25,"alien 2",fontsize=font_size1, color='y',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien2_energy_label = r.axes.text(x[0,5],x[1,5]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien2_hp_label = r.axes.text(x[0,5],x[1,5]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    alien3_label = r.axes.text(x[0,6],x[1,6]+0.25,"alien 3",fontsize=font_size1, color='y',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien3_energy_label = r.axes.text(x[0,6],x[1,6]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien3_hp_label = r.axes.text(x[0,6],x[1,6]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    r.step()

    for t in range(iterations):
        # Get the most recent pose information from the Robotarium. The time delay is
        # approximately 0.033s
        x = r.get_poses()
        xi = uni_to_si_states(x)
        avoidpoints = np.array([[random.uniform(-0.4, -0.4)], [random.uniform(-1, 1)]])

        # system parameters
        explorer_system_energy_cost = 400
        explorer_system_hp_cost = 400

        explorer1_label.set_position([xi[0,0],xi[1,0]+0.25])
        explorer1_label.set_fontsize(determine_font_size(r,font_size_m1))
        explorer1_energy_label.set_position([xi[0,0],xi[1,0]+0.2])
        explorer1_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer1_energy_label.set_text("NRG: " + str(round(agent_energy_level[0], 2)))
        explorer1_hp_label.set_position([xi[0,0],xi[1,0]+0.15])
        explorer1_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer1_hp_label.set_text("HP: " + str(round(agent_hp_level[0], 2)))

        explorer2_label.set_position([xi[0,1],xi[1,1]+0.25])
        explorer2_label.set_fontsize(determine_font_size(r,font_size_m1))
        explorer2_energy_label.set_position([xi[0,1],xi[1,1]+0.2])
        explorer2_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer2_energy_label.set_text("NRG: " + str(round(agent_energy_level[1], 2)))
        explorer2_hp_label.set_position([xi[0,1],xi[1,1]+0.15])
        explorer2_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer2_hp_label.set_text("HP: " + str(round(agent_hp_level[1], 2)))

        explorer3_label.set_position([xi[0,2],xi[1,2]+0.25])
        explorer3_label.set_fontsize(determine_font_size(r,font_size_m1))
        explorer3_energy_label.set_position([xi[0,2],xi[1,2]+0.2])
        explorer3_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer3_energy_label.set_text("NRG: " + str(round(agent_energy_level[2], 2)))
        explorer3_hp_label.set_position([xi[0,2],xi[1,2]+0.15])
        explorer3_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer3_hp_label.set_text("HP: " + str(round(agent_hp_level[2], 2)))

        explorer4_label.set_position([xi[0,3],xi[1,3]+0.25])
        explorer4_label.set_fontsize(determine_font_size(r,font_size_m1))
        explorer4_energy_label.set_position([xi[0,3],xi[1,3]+0.2])
        explorer4_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer4_energy_label.set_text("NRG: " + str(round(agent_energy_level[3], 2)))
        explorer4_hp_label.set_position([xi[0,3],xi[1,3]+0.15])
        explorer4_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer4_hp_label.set_text("HP: " + str(round(agent_hp_level[3], 2)))

        alien1_label.set_position([xi[0,4],xi[1,4]+0.25])
        alien1_label.set_fontsize(determine_font_size(r,font_size_m1))
        alien1_energy_label.set_position([xi[0,4],xi[1,4]+0.2])
        alien1_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien1_energy_label.set_text("NRG: " + str(round(agent_energy_level[4], 2)))
        alien1_hp_label.set_position([xi[0,4],xi[1,4]+0.15])
        alien1_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien1_hp_label.set_text("HP: " + str(round(agent_hp_level[4], 2)))

        alien2_label.set_position([xi[0,5],xi[1,5]+0.25])
        alien2_label.set_fontsize(determine_font_size(r,font_size_m1))
        alien2_energy_label.set_position([xi[0,5],xi[1,5]+0.2])
        alien2_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien2_energy_label.set_text("NRG: " + str(round(agent_energy_level[5], 2)))
        alien2_hp_label.set_position([xi[0,5],xi[1,5]+0.15])
        alien2_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien2_hp_label.set_text("HP: " + str(round(agent_hp_level[5], 2)))

        alien3_label.set_position([xi[0,6],xi[1,6]+0.25])
        alien3_label.set_fontsize(determine_font_size(r,font_size_m1))
        alien3_energy_label.set_position([xi[0,6],xi[1,6]+0.2])
        alien3_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien3_energy_label.set_text("NRG: " + str(round(agent_energy_level[6], 2)))
        alien3_hp_label.set_position([xi[0,6],xi[1,6]+0.15])
        alien3_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien3_hp_label.set_text("HP: " + str(round(agent_hp_level[6], 2)))

        # This updates the marker sizes if the figure window size is changed. 
        # This should be removed when submitting to the Robotarium.
        waypoint_text[0].set_fontsize(determine_font_size(r,font_size_m))
        waypoint_text[1].set_fontsize(determine_font_size(r,0.1))
        waypoint_text[2].set_fontsize(determine_font_size(r,0.06))
        g[0].set_sizes([determine_marker_size(r,0.5)])
        g[1].set_sizes([determine_marker_size(r,0.4)])
        g[2].set_sizes([determine_marker_size(r,0.2)])

        obstaclepoint_text[0].set_fontsize(determine_font_size(r,font_size_m))
        g1[0].set_sizes([determine_marker_size(r,0.5)])
        g1[1].set_sizes([determine_marker_size(r,0.3)])

        L = cycle_GL(N)

        for i in range(N):
            # k = topological_neighbors(L, i)
            k = [4, 5, 6]
            u = [0, 1, 2, 3]
            strategy = []
            # if !len(activeAlien):
            if i < 4:
                for j in k:
                    if np.linalg.norm(x[:2,[i]] - x[:2,[j]]) <= sensing_distance:
                        goalpoint = waypoints[:,stateList[i]].reshape((2,1))
                        # strategy = greedy_strategy_selector(1, x[:2, i], goalpoint)
                        strategy = greedy_strategy_selector(1, 1, agent_energy_level, agent_hp_level)
                        # print(str(i) + str(strategy[1]))

                        # # if t%20 == 0:
                        # #     print(111111)
                        # if strategy[1] == 'change_direction' and t%20 == 0:
                        #     print(111111)
                        #     si_velocities[:,i] = np.sum(avoidpoints[:, 0, None] - xi[:, i, None], 1)
                        # elif strategy[1] == 'change_speed' and t%20 == 0:                                   
                        #     print(str(i) + str(state1))
                        #     if np.linalg.norm(x[:2,[i]] - goalpoint) < close_enough:
                        #         state1 += 1
                        #         # print('1' + str(state1))

                        #     if agent_hp_level[i] > 0:
                        #         si_velocities[:,i] = np.sum(goalpoint[:, 0, None] - xi[:, i, None], 1)
                        #     else:
                        #         si_velocities[:,i] = np.sum(xi[:, i, None] - xi[:, i, None], 1)

                        if t%20 == 0:
                            if strategy[1] == 'change_direction':
                                si_velocities[:,i] = np.sum(avoidpoints[:, 0, None] - xi[:, i, None], 1)
                            elif strategy[1] == 'change_speed':                                   
                                # print(str(i) + str(state1))
                                # if np.linalg.norm(x[:2,[i]] - goalpoint) < close_enough:
                                #     if stateList[i] < 2:
                                #         stateList[i] += stateList[i] + 1
                                #     else:
                                #         stateList[i] = 2
                                    # print('1' + str(state1))

                                if agent_hp_level[i] > 0:
                                    si_velocities[:,i] = np.sum(goalpoint[:, 0, None] - xi[:, i, None], 1)
                                else:
                                    si_velocities[:,i] = np.sum(xi[:, i, None] - xi[:, i, None], 1)

                    elif np.linalg.norm(x[:2,[i]] - x[:2,[j]]) > sensing_distance:
                        waypoint = waypoints[:,stateList[i]].reshape((2,1))

                        if agent_hp_level[i] > 0:
                            si_velocities[:,[i]] = explorer_controller(x[:2,[i]], waypoint)
                        else:
                            si_velocities[:,[i]] = explorer_controller(x[:2,[i]], x[:2,[i]])

                        if np.linalg.norm(x[:2,[i]] - waypoint) < close_enough:
                            # state = (state + 1)%4
                            if stateList[i] < 2:
                                stateList[i] += 1
                            else:
                                stateList[i] = 2

                        print(str(i) + ' ' +str(stateList[i]))
            elif i >= 4:
                for j in u:
                    if j < 4 and np.linalg.norm(x[:2,[i]] - x[:2,[j]]) <= sensing_distance:
                        si_velocities[:, i] = np.sum(xi[:, j] - xi[:, i, None], 1)
                    elif np.linalg.norm(x[:2,[i]] - x[:2,[j]]) > sensing_distance:
                        if i == 4:
                            alienpoint3 = alienpoints3[:,alienState].reshape((2,1))
                            si_velocities[:,[N-3]] = alien2_controller(x[:2,[N-3]], alienpoint3)
                        elif i == 5:
                            alienpoint2 = alienpoints2[:,alienState].reshape((2,1))
                            si_velocities[:,[N-2]] = alien2_controller(x[:2,[N-2]], alienpoint2)
                        elif i == 6:
                            alienpoint1 = alienpoints1[:,alienState].reshape((2,1))
                            si_velocities[:,[N-1]] = alien1_controller(x[:2,[N-1]], alienpoint1)

                tmppoint = waypoints[:,alienState].reshape((2,1))

                if np.linalg.norm(x[:2,[0]] - tmppoint) < close_enough:
                    if alienState < 1:
                        # alienState = (alienState + 1)%2
                        alienState += 1
                    else:
                        alienState = 0

            # #Keep single integrator control vectors under specified magnitude
            # # Threshold control inputs
            # norms = np.linalg.norm(dxi, 2, 0)
            # idxs_to_normalize = (norms > magnitude_limit)
            # dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

            # dxi = si_barrier_cert(si_velocities, xi)
            dxi = si_barrier_cert(si_velocities, x[:2,:])
            dxu = si_to_uni_dyn(dxi,x)

            # Set the velocities of agents 1,...,N to dxu
            r.set_velocities(np.arange(N), dxu)

        # Calculate agent energy cost
        for i in range(N):
            agent_energy_level[i] -= np.linalg.norm(old_x[i] - x[:2,[i]]) * 10

        # Calculate agent hp cost
        for i in range(N-3):
            if agent_hp_level[i] > 0:
                if np.linalg.norm(x[:2,[i]] - x[:2,[-1]]) < sensing_distance:
                    # numActiveAlien += 1
                    agent_hp_level[i] -= unit_alien_attacking_hp_cost;
                    agent_hp_level[-1] -= unit_explorer_attacking_hp_cost;

                if np.linalg.norm(x[:2,[i]] - x[:2,[-2]]) < sensing_distance:
                    # numActiveAlien += 1
                    agent_hp_level[i] -= unit_alien_attacking_hp_cost;
                    agent_hp_level[-2] -= unit_explorer_attacking_hp_cost;

                if np.linalg.norm(x[:2,[i]] - x[:2,[-3]]) < sensing_distance:
                    # numActiveAlien += 1
                    agent_hp_level[i] -= unit_alien_attacking_hp_cost;
                    agent_hp_level[-3] -= unit_explorer_attacking_hp_cost;

        numActiveAlien = 0
        # detect the number of aliens
        if alien_detector(N-3, x, -1, sensing_distance):
            activeAlien.append(6)
            numActiveAlien +=1

        if alien_detector(N-3, x, -2, sensing_distance):
            activeAlien.append(5)
            numActiveAlien +=1

        if alien_detector(N-3, x, -3, sensing_distance):
            activeAlien.append(4)
            numActiveAlien +=1

        # recode old position
        old_x.clear()

        for i in range(N):
            old_x.append(x[:2, [i]])  

        m = 0

        for i in range(N):
            if i < 4 and agent_hp_level[i] <= 0:
                m += 1

        tmpe = 0
        tmphp = 0

        for i in range(N):
            if i < 4:
                tmpe += agent_energy_level[i]
                tmphp += agent_hp_level[i]

        explorer_system_energy_cost -= tmpe
        explorer_system_hp_cost -= tmphp

        print(explorer_system_energy_cost)
        print(explorer_system_hp_cost)
        print('cost time is ' + str(t * 0.033))

        if m == 4:
            tmpe = 0
            tmphp = 0

            for i in range(N):
                if i < 4:
                    tmpe += agent_energy_level[i]
                    tmphp += agent_hp_level[i]

            explorer_system_energy_cost -= tmpe
            explorer_system_hp_cost -= tmphp

            print(explorer_system_energy_cost)
            print(explorer_system_hp_cost)
            print('cost time is ' + str(k * 0.033))

            os._exit(0)    

        # Iterate the simulation
        r.step()

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

def main():
    gut_explore_game()

if __name__ == '__main__':
    main()