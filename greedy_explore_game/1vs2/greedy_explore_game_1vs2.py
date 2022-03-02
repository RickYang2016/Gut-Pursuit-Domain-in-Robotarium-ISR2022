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
import greedy_single2


def normal_situation(N, L, situation):
    if situation == 'patroling':
        agents = -lineGL(N)
        L[0:N,0:N] = agents
    # elif situation == 'attacking':
    #     agents = -completeGL(N)
    #     L[0:N,0:N] = agents
    # elif situation == 'defending':
    #     # agents = -completeGL(N)
    #     # L[0:N,0:N] = agents
    #     agents = -completeGL(N-1)
    #     L[1:N,1:N] = agents
    #     L[1,1] = L[1,1] + 1
    #     L[1,0] = -1

    return L

def greedy_strategy_selector(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level):
    strategy = []

    tmp = sorted(greedy_single2.Greedy_DecisionMaking(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level).items(), key = lambda kv:(kv[1], kv[0]))[-1]
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

# 1 explorer vs 1 alien
def gut_explore_game():
    # Experiment Constants
    iterations = 5000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
    N=3 #Number of robots to use, this must stay 4 unless the Laplacian is changed.

    # waypoints = np.array([[-0.36, 0.66, 1.158],[-0.36, -0.66, 0.8]]) #Waypoints the leader moves to.
    waypoints = np.array([[-0.36, -0.13, 1.158],[-0.36, -0.1, 0.8]]) #Waypoints the leader moves to.
    # waypoints = np.array([[-0.36, -0.361, 1.158],[-0.36, -0.1, 0.8]]) #Waypoints the leader moves to.

    alienpoints1 = np.array([[-0.5, 0.296, 0.296],[-0.1, -0.01, -0.01]]) # attacking
    alienpoints2 = np.array([[-0.13, -0.291, -0.165],[-0.161, 0.01, 0.056]]) # attacking
    # alienpoints1 = np.array([[-0.591, 0.12, 0.471],[-0.1, -0.76, -0.571]]) # attacking
    # alienpoints2 = np.array([[-0.13, 0.14, 0.421],[-0.161, -0.515, -0.239]]) # attacking
    close_enough = 0.03; #How close the leader must get to the waypoint to move to the next one.

    # sensing distance between explorer and alien
    sensing_distance = 0.6

    # For computational/memory reasons, initialize the velocity vector
    dxi = np.zeros((2,N))

    #Initialize leader state
    state = 0
    alienState = 0
    numActiveAlien = 0

    unit_explorer_attacking_hp_cost = 0.03
    unit_alien_attacking_hp_cost = 0.05

    # initial agent's energy and hp
    agent_energy_level = []
    agent_hp_level = []

    for i in range(N):
        if i < N-2:
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
    # initial_conditions = np.array([[-0.682, 0.692, -0.8],[-0.787, 0.121, 0.463],[0, 0, 0]])
    initial_conditions = np.array([[-0.682, 0.707, -0.8],[-0.787, 0.194, 0.463],[0, 0, 0]])

    # Instantiate the Robotarium object with these parameters
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

    # Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
    # Single-integrator -> unicycle dynamics mapping
    _,uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    # Single-integrator barrier certificates
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    # Single-integrator position controller
    leader_controller = create_si_position_controller(velocity_magnitude_limit=0.15)
    alien1_controller = create_si_position_controller(velocity_magnitude_limit=0.09)
    alien2_controller = create_si_position_controller(velocity_magnitude_limit=0.09)

    # define x initially
    m = np.zeros((2,N))

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

    # Plot Graph Connections
    x = r.get_poses() # Need robot positions to do this.
    old_x = []

    # Create goal text and markers

    # Define goal points by removing orientation from poses
    goal_points = generate_initial_conditions(N, width=r.boundaries[2]-2*r.robot_diameter, height = r.boundaries[3]-2*r.robot_diameter, spacing=0.5)

    si_velocities = np.zeros((2, N))

    #Text with goal identification
    goal_caption = ['Treasure']

    #Plot text for caption
    waypoint_text = r.axes.text(waypoints[0, 2], waypoints[1, 2], goal_caption[0], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
    g = r.axes.scatter(waypoints[0, 2], waypoints[1, 2], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM1[1,:],linewidth=line_width,zorder=-2)
    # robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM2[ii,:],linewidth=line_width) 
    # for ii in range(goal_points.shape[1])]

    for i in range(N):
        old_x.append(initial_conditions[:2, [i]])


    explorer_label = r.axes.text(x[0,0],x[1,0]+0.25,"explorer",fontsize=font_size1, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer_energy_label = r.axes.text(x[0,0],x[1,0]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    explorer_hp_label = r.axes.text(x[0,0],x[1,0]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    alien1_label = r.axes.text(x[0,1],x[1,1]+0.25,"alien 1",fontsize=font_size1, color='r',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien1_energy_label = r.axes.text(x[0,1],x[1,1]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien1_hp_label = r.axes.text(x[0,1],x[1,1]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    alien2_label = r.axes.text(x[0,2],x[1,2]+0.25,"alien 2",fontsize=font_size1, color='r',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien2_energy_label = r.axes.text(x[0,2],x[1,2]+0.2,"NRG: ",fontsize=font_size2, color='c',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
    alien2_hp_label = r.axes.text(x[0,2],x[1,2]+0.15,"HP: ",fontsize=font_size2, color='m',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

    r.step()

    for t in range(iterations):

        # Get the most recent pose information from the Robotarium. The time delay is
        # approximately 0.033s
        x = r.get_poses()
        xi = uni_to_si_states(x)
        # L = np.zeros((N,N))

        # system parameters
        explorer_system_energy_cost = 100
        explorer_system_hp_cost = 100
        
        avoidpoints = np.array([[random.uniform(-1.4, 1.4)], [random.uniform(-1.4, 1.4)]])

        explorer_label.set_position([xi[0,0],xi[1,0]+0.25])
        explorer_label.set_fontsize(determine_font_size(r,font_size_m1))
        explorer_energy_label.set_position([xi[0,0],xi[1,0]+0.2])
        explorer_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer_energy_label.set_text("NRG: " + str(round(agent_energy_level[0], 2)))
        explorer_hp_label.set_position([xi[0,0],xi[1,0]+0.15])
        explorer_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        explorer_hp_label.set_text("HP: " + str(round(agent_hp_level[0], 2)))

        alien1_label.set_position([xi[0,1],xi[1,1]+0.25])
        alien1_label.set_fontsize(determine_font_size(r,font_size_m1))
        alien1_energy_label.set_position([xi[0,1],xi[1,1]+0.2])
        alien1_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien1_energy_label.set_text("NRG: " + str(round(agent_energy_level[1], 2)))
        alien1_hp_label.set_position([xi[0,1],xi[1,1]+0.15])
        alien1_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien1_hp_label.set_text("HP: " + str(round(agent_hp_level[1], 2)))

        alien2_label.set_position([xi[0,2],xi[1,2]+0.25])
        alien2_label.set_fontsize(determine_font_size(r,font_size_m1))
        alien2_energy_label.set_position([xi[0,2],xi[1,2]+0.2])
        alien2_energy_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien2_energy_label.set_text("NRG: " + str(round(agent_energy_level[2], 2)))
        alien2_hp_label.set_position([xi[0,2],xi[1,2]+0.15])
        alien2_hp_label.set_fontsize(determine_font_size(r,font_size_m2))
        alien2_hp_label.set_text("HP: " + str(round(agent_hp_level[2], 2)))

        # This updates the marker sizes if the figure window size is changed. 
        # This should be removed when submitting to the Robotarium.
        # waypoint_text[0].set_fontsize(determine_font_size(r,font_size_m))
        waypoint_text.set_fontsize(determine_font_size(r,0.06))
        g.set_sizes([determine_marker_size(r,0.2)])
        # # Update Robot Marker Plotted Visualization
        # for i in range(x.shape[1]):
        #     robot_markers[i].set_offsets(x[:2,i].T)
        #     # This updates the marker sizes if the figure window size is changed. 
        #     # This should be removed when submitting to the Robotarium.
        #     robot_markers[i].set_sizes([determine_marker_size(r, robot_marker_size_m)])

        if numActiveAlien == 0:
            L = np.zeros((N-1,N-1))
            L = normal_situation(N-1, L, 'patroling')

            #Leader
            waypoint = waypoints[:,state].reshape((2,1))

            #Alien
            alienpoint1 = alienpoints1[:,state].reshape((2,1))
            alienpoint2 = alienpoints2[:,state].reshape((2,1))

            dxi[:,[N-1]] = alien1_controller(x[:2,[N-1]], alienpoint1)
            dxi[:,[N-2]] = alien2_controller(x[:2,[N-2]], alienpoint2)
            dxi[:,[0]] = leader_controller(x[:2,[0]], waypoint)

            if np.linalg.norm(x[:2,[0]] - waypoint) < close_enough:
                state = (state + 1)%4

            #Keep single integrator control vectors under specified magnitude
            # Threshold control inputs
            norms = np.linalg.norm(dxi, 2, 0)
            idxs_to_normalize = (norms > magnitude_limit)
            dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

            #Use barriers and convert single-integrator to unicycle commands
            dxi = si_barrier_cert(dxi, x[:2,:])
            dxu = si_to_uni_dyn(dxi,x)

            # Set the velocities of agents 1,...,N to dxu
            r.set_velocities(np.arange(N), dxu)

        else:
            goalPos = np.array([[1.158],[0.8]])
            strategy = []
            for i in range(N):
                if i == 0:
                    strategy = greedy_strategy_selector(N-2, numActiveAlien, agent_energy_level, agent_hp_level)
                    print(strategy)

            L = cycle_GL(N)

            for i in range(N):
                # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
                j = topological_neighbors(L, i)
                # Compute the pp algorithm
                if i == 0 and t%20 == 0:
                    if strategy[1] == 'change_direction':
                        si_velocities[:,i] = np.sum(avoidpoints[:, 0, None] - xi[:, i, None], 1)
                    elif strategy[1] == 'change_speed':
                        si_velocities[:,i] = np.sum(goalPos[:, 0, None] - xi[:, i, None], 1)
                if i >= 1:
                    si_velocities[:, i] = np.sum(xi[:, j] - xi[:, i, None], 1)

            # # #Keep single integrator control vectors under specified magnitude
            # # # Threshold control inputs
            # norms = np.linalg.norm(si_velocities, 2, 0)
            # idxs_to_normalize = (norms > magnitude_limit)
            # si_velocities[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

            # Use the barrier certificate to avoid collisions
            si_velocities = si_barrier_cert(si_velocities, xi)

            # Transform single integrator to unicycle
            dxu = si_to_uni_dyn(si_velocities, x)

            for i in range(N):
                if i  == 0 and strategy[1] == 'change_speed': # explorer
                    dxu[:,i] = dxu[:,i] * 1.05
                # if i >= 1: # aliens
                #     dxu[:,i] = dxu[:,i] * 1.05

            # Set the velocities of agents 1,...,N
            r.set_velocities(np.arange(N), dxu)


        # print(numActiveAlien)
        numActiveAlien = 0

        # Calculate agent energy cost
        for i in range(N):
            agent_energy_level[i] -= np.linalg.norm(old_x[i] - x[:2,[i]]) * 10

        # Calculate agent hp cost
        for i in range(N-2):
            if np.linalg.norm(x[:2,[i]] - x[:2,[-1]]) < sensing_distance:
                # numActiveAlien += 1
                agent_hp_level[i] -= unit_alien_attacking_hp_cost;
                agent_hp_level[-1] -= unit_explorer_attacking_hp_cost;

            if np.linalg.norm(x[:2,[i]] - x[:2,[-2]]) < sensing_distance:
                # numActiveAlien += 1
                agent_hp_level[i] -= unit_alien_attacking_hp_cost;
                agent_hp_level[-2] -= unit_explorer_attacking_hp_cost;

        # detect the number of aliens
        if alien_detector(N-2, x, -1, sensing_distance):
            numActiveAlien +=1

        if alien_detector(N-2, x, -2, sensing_distance):
            numActiveAlien +=1

        # recode old position
        old_x.clear()

        for i in range(N):
            old_x.append(x[:2, [i]])

        # if agent_hp_level[0] <= 0:
        if agent_hp_level[0] <= 0 or (np.linalg.norm(x[:2,[0]] - waypoints[:2, -1]) < 0.65):
            tmpe = 0
            tmphp = 0

            for i in range(N):
                if i < 1:
                    tmpe += agent_energy_level[i]
                    tmphp += agent_hp_level[i]

            explorer_system_energy_cost -= tmpe
            explorer_system_hp_cost -= tmphp

            print(explorer_system_energy_cost)
            print(explorer_system_hp_cost)
            print('cost time is ' + str(t * 0.033))

            os._exit(0)

        # Iterate the simulation
        r.step()

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

def main():
    gut_explore_game()

if __name__ == '__main__':
    main()