'''
Author: Qin Yang
05/08/2021
'''

import numpy as np
from fractions import Fraction
import time
import math
import scipy.stats as st


def first_level_mne(e, a, strategyProbablity):
    # for s in range(len(strategyProbablity)):
    #     if s == 0:
    #         e['attacking'] = strategyProbablity[s]
    #     elif s == 1:
    #         e['defending'] = strategyProbablity[s]
    #     elif s == 2:
    #         a['attacking'] = strategyProbablity[s]
    #     elif s == 3:
    #         a['defending'] = strategyProbablity[s]

    for s in range(len(strategyProbablity)):
        if s == 0:
            e['defending'] = strategyProbablity[s]
        elif s == 1:
            e['attacking'] = strategyProbablity[s]
        elif s == 2:
            a['attacking'] = strategyProbablity[s]
        elif s == 3:
            a['defending'] = strategyProbablity[s]

    print(e)

    return (e, a)


def second_level_mne(e, a, strategyProbablity):
    for s in range(len(strategyProbablity)):
        if s == 0:
            e['change_speed'] = strategyProbablity[s]
        elif s ==1:
            e['change_direction'] = strategyProbablity[s]
        elif s == 2:
            a['follow'] = strategyProbablity[s]
        elif s == 3:
            a['back'] = strategyProbablity[s]

    # for s in range(len(strategyProbablity)):
    #     if s == 0:
    #         e['change_direction'] = strategyProbablity[s]
    #     elif s ==1:
    #         e['change_speed'] = strategyProbablity[s]
    #     elif s == 2:
    #         a['back'] = strategyProbablity[s]
    #     elif s == 3:
    #         a['follow'] = strategyProbablity[s]

    return (e, a)

def GUT_CPT(e1_first_level_strategy, a1_first_level_strategy, second_level_mne, agent2_strategy):
    result = 0

    for item in second_level_mne.items():
        if item[0] == agent2_strategy:
            result = Fraction(e1_first_level_strategy) * Fraction(a1_first_level_strategy) * Fraction(item[1])
        
    return result

def final_explorer_strategy_combination(explorer_gut_cpt):
    tmp = sorted(explorer_gut_cpt.items(), key = lambda kv:(kv[1], kv[0]))[-1]

    if tmp[0] == 'e11a11e21t' or tmp[0] == 'e11a12e22t':
        return('attacking', 'change_speed')
    elif tmp[0] == 'e11a11e21l' or tmp[0] == 'e11a12e22l':
        return('attacking', 'change_direction')
    elif tmp[0] == 'e12a11e23t' or tmp[0] == 'e12a12e24t':
        return('defending', 'change_speed')
    elif tmp[0] == 'e12a11e23l' or tmp[0] == 'e12a12e24l':
        return('defending', 'change_direction')

def final_alien_strategy_combination(alien_gut_cpt):
    tmp = sorted(alien_gut_cpt.items(), key = lambda kv:(kv[1], kv[0]))[-1]

    if tmp[0] == 'e11a11a21t' or tmp[0] == 'e11a12a22t':
        return('attacking', 'follow')
    elif tmp[0] == 'e11a11a21l' or tmp[0] == 'e11a12a22l':
        return('attacking', 'back')
    elif tmp[0] == 'e12a11a23t' or tmp[0] == 'e12a12a24t':
        return('defending', 'follow')
    elif tmp[0] == 'e12a11a23l' or tmp[0] == 'e12a12a24l':
        return('defending', 'back')


def explorerAbility(explorersEnergyLevel, explorersHPLevel):
    a = 0.0111
    d = 0.0222

    explorerAttackingAbility = a * explorersEnergyLevel
    explorerDefendingAbility = d * explorersHPLevel

    return(explorerAttackingAbility, explorerDefendingAbility)

def alienAbility(aliensEnergyLevel, aliensHPLevel):
    a = 0.0107
    d = 0.0143

    alienAttackingAbility = a * aliensEnergyLevel
    alienDefendingAbility = d * aliensHPLevel

    return(alienAttackingAbility, alienDefendingAbility)


def WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, situation):
    explorers_energy_level = []
    explorers_HP_level = []
    aliens_energy_level = []
    aliens_HP_level = []
    attackingSituationCoefficient1 = 1.3
    attackingSituationCoefficient2 = 0.7
    defendingSituationCoefficient1 = 0.6
    defendingSituationCoefficient2 = 1.4
    # print(numActiveAlien)

    for i in range(numExplorer):
        explorers_energy_level.append(agent_energy_level[i])
        explorers_HP_level.append(agent_hp_level[i])

    aliens_energy_level = list(set(agent_energy_level).difference(set(explorers_energy_level)))
    aliens_HP_level = list(set(agent_hp_level).difference(set(explorers_HP_level)))

    explorerAttackingAbility, explorerDefendingAbility = explorerAbility(sum(explorers_energy_level), sum(explorers_HP_level))
    alienAttackingAbility, alienDefendingAbility = alienAbility(sum(aliens_energy_level), sum(aliens_HP_level))

    if situation == "10":
        winningUtility = math.pow((attackingSituationCoefficient1 * explorerAttackingAbility + attackingSituationCoefficient2 * explorerDefendingAbility) 
                                / (defendingSituationCoefficient1 * alienAttackingAbility + defendingSituationCoefficient2 * alienDefendingAbility), numExplorer / numActiveAlien)
    elif situation == "11":
        winningUtility = math.pow((attackingSituationCoefficient1 * explorerAttackingAbility + attackingSituationCoefficient2 * explorerDefendingAbility) 
                                / (attackingSituationCoefficient1 * alienAttackingAbility + attackingSituationCoefficient2 * alienDefendingAbility), numExplorer / numActiveAlien)
    elif situation == "01":
        winningUtility = math.pow((defendingSituationCoefficient1 * explorerAttackingAbility + defendingSituationCoefficient2 * explorerDefendingAbility) 
                                / (attackingSituationCoefficient1 * alienAttackingAbility + attackingSituationCoefficient2 * alienDefendingAbility), numExplorer / numActiveAlien)
    elif situation == "00":
        winningUtility = math.pow((defendingSituationCoefficient1 * explorerAttackingAbility + defendingSituationCoefficient2 * explorerDefendingAbility) 
                                / (defendingSituationCoefficient1 * alienAttackingAbility + defendingSituationCoefficient2 * alienDefendingAbility), numExplorer / numActiveAlien)

    return winningUtility

def HPUtility(numAttackingAgent, numAttackingAdversary, agent_hp_level, explorerStrategy, alienStrategy):
    explorers_HP_level = []
    aliens_HP_level = []
    agentRadius = 1
    agentArea = math.pow(agentRadius, 2) * math.pi
    explorerC = 0
    alienC = 0
    tmp1 = 0
    tmp2 = 0

    if explorerStrategy == 'change_speed' and alienStrategy == 'follow':
        explorerC = agentArea * 1.5
        alienC = agentArea * 1.2
    elif explorerStrategy == 'change_speed' and alienStrategy == 'back':
        explorerC = agentArea * 1.5
        alienC = agentArea * 0.8
    elif explorerStrategy == 'change_direction' and alienStrategy == 'follow':
        explorerC = agentArea * 0.8
        alienC = agentArea * 1.2
    elif explorerStrategy == 'change_direction' and alienStrategy == 'back':
        explorerC = agentArea * 0.8
        alienC = agentArea * 0.8

    for i in range(numAttackingAgent):
        explorers_HP_level.append(agent_hp_level[i])

    aliens_HP_level = list(set(agent_hp_level).difference(set(explorers_HP_level)))

    # for i in range(99):
    #     tmp1 = tmp1 + st.poisson(alienC).pmf(i) * np.mean(explorers_HP_level) * i
    #     tmp2 = tmp2 + st.poisson(explorerC).pmf(i) * np.mean(aliens_HP_level) * i

    # result = abs(numAttackingAgent * tmp1 - numAttackingAdversary * tmp2)

    # result = abs(numAttackingAgent * alienC * np.mean(explorers_HP_level) - numAttackingAdversary * explorerC * np.mean(aliens_HP_level))
    result = -numAttackingAgent * alienC * np.mean(explorers_HP_level) + numAttackingAdversary * explorerC * np.mean(aliens_HP_level)

    return result

def Greedy_DecisionMaking(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level):
    m11 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "10") * 100)
    m12 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "11") * 100)
    m21 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "01") * 100)
    m22 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "00") * 100)


    x11 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_speed', 'follow'))
    x12 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_speed', 'back'))
    x21 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_direction', 'follow'))
    x22 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_direction', 'back'))

    reward = {}

    reward['ad'] = (m11 + m12) * (x21 + x22) 
    reward['as'] = (m11 + m12) * (x11 + x12) * 1.2
    reward['dd'] = (m21 + m22) * (x21 + x22) * 7.5
    reward['ds'] = (m21 + m22) * (x11 + x12) * 1.5

    # reward['ad'] = (m11 + m12) * (x21 + x22) * 1
    # reward['as'] = (m11 + m12) * (x11 + x12) * 0.6
    # reward['dd'] = (m21 + m22) * (x21 + x22) * 1.95
    # reward['ds'] = (m21 + m22) * (x11 + x12) * 0.83

    # reward['ad'] = (m11 + m12) * (x21 + x22) 
    # reward['as'] = (m11 + m12) * (x11 + x12)
    # reward['dd'] = (m21 + m22) * (x21 + x22)
    # reward['ds'] = (m21 + m22) * (x11 + x12)

    # print(reward)
    
    return reward