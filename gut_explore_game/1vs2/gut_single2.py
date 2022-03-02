'''
Author: Qin Yang
05/08/2021
'''

import numpy as np
from fractions import Fraction
import time
import nashlh
import math
import scipy.stats as st


def first_level_mne(e, a, strategyProbablity):
    for s in range(len(strategyProbablity)):
        if s == 0:
            e['attacking'] = strategyProbablity[s]
        elif s == 1:
            e['defending'] = strategyProbablity[s]
        elif s == 2:
            a['defending'] = strategyProbablity[s]
        elif s == 3:
            a['attacking'] = strategyProbablity[s]

    print(e)

    return (e, a)


def second_level_mne(e, a, strategyProbablity):
    for s in range(len(strategyProbablity)):
        if s == 0:
            e['change_direction'] = strategyProbablity[s]
        elif s ==1:
            e['change_speed'] = strategyProbablity[s]
        elif s == 2:
            a['follow'] = strategyProbablity[s]
        elif s == 3:
            a['back'] = strategyProbablity[s]

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
        # return('attacking', 'follow')
        return('defending', 'follow')
    elif tmp[0] == 'e11a11a21l' or tmp[0] == 'e11a12a22l':
        return('attacking', 'back')
    elif tmp[0] == 'e12a11a23t' or tmp[0] == 'e12a12a24t':
        # return('defending', 'follow')
        return('attacking', 'follow')
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
    explorerArea = 0
    alienArea = 0
    tmp1 = 0
    tmp2 = 0

    if explorerStrategy == 'change_speed' and alienStrategy == 'follow':
        explorerArea = agentArea * 1.5
        alienArea = agentArea * 1.2
    elif explorerStrategy == 'change_speed' and alienStrategy == 'back':
        explorerArea = agentArea * 1.5
        alienArea = agentArea * 0.8
    elif explorerStrategy == 'change_direction' and alienStrategy == 'follow':
        explorerArea = agentArea * 0.8
        alienArea = agentArea * 1.2
    elif explorerStrategy == 'change_direction' and alienStrategy == 'back':
        explorerArea = agentArea * 0.8
        alienArea = agentArea * 0.8

    for i in range(numAttackingAgent):
        explorers_HP_level.append(agent_hp_level[i])

    aliens_HP_level = list(set(agent_hp_level).difference(set(explorers_HP_level)))

    # for i in range(99):
    #     tmp1 = tmp1 + st.poisson(alienArea).pmf(i) * np.mean(explorers_HP_level) * i
    #     tmp2 = tmp2 + st.poisson(explorerArea).pmf(i) * np.mean(aliens_HP_level) * i

    # result = abs(numAttackingAgent * tmp1 - numAttackingAdversary * tmp2)

    result = abs(numAttackingAgent * alienArea * np.mean(explorers_HP_level) - numAttackingAdversary * explorerArea * np.mean(aliens_HP_level))

    return result

def GUT_DecisionMaking(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level):
    # Building Attacking GUT Structure
    # ============================================================ Utility Matrix ============================================
    # Level 1 What -- attacking or defending (attacking) velocity selection
    m11 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "10") * 100)
    m12 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "11") * 100)
    m21 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "01") * 100)
    m22 = round(WinningUtility(numExplorer, numActiveAlien, agent_energy_level, agent_hp_level, "00") * 100)

    n11 = round(WinningUtility(numActiveAlien, numExplorer, agent_energy_level, agent_hp_level, "10") * 100)
    n12 = round(WinningUtility(numActiveAlien, numExplorer, agent_energy_level, agent_hp_level, "11") * 100)
    n21 = round(WinningUtility(numActiveAlien, numExplorer, agent_energy_level, agent_hp_level, "01") * 100)
    n22 = round(WinningUtility(numActiveAlien, numExplorer, agent_energy_level, agent_hp_level, "00") * 100)  

    # situationMatrixLevel1 = ['2 1\n4 3', '1 4\n4 1\n']
    # situationMatrixLevel1 = ['1 1\n1 11', '1 1\n1 11\n']
    situationMatrixLevel1 = [str(m11) + ' ' + str(m12) + '\n' + str(m21) + ' ' + str(m22), str(n11) + ' ' + str(n12) + '\n' + str(n21) + ' ' + str(n22) + '\n']

    x11 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_speed', 'follow'))
    x12 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_speed', 'back'))
    x21 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_direction', 'follow'))
    x22 = round(HPUtility(numExplorer, numActiveAlien, agent_hp_level, 'change_direction', 'back'))

    y11 = round(HPUtility(numActiveAlien, numExplorer, agent_hp_level, 'change_speed', 'follow'))
    y12 = round(HPUtility(numActiveAlien, numExplorer, agent_hp_level, 'change_speed', 'back'))
    y21 = round(HPUtility(numActiveAlien, numExplorer, agent_hp_level, 'change_direction', 'follow'))
    y22 = round(HPUtility(numActiveAlien, numExplorer, agent_hp_level, 'change_direction', 'back'))

    # Level 2 How
    situationMatrixLevel2 = [str(x11) + ' ' + str(x12) + '\n' + str(x21) + ' ' + str(x22), str(y11) + ' ' + str(y12) + '\n' + str(y21) + ' ' + str(y22) + '\n']

    print(situationMatrixLevel1)
    print(situationMatrixLevel2)
    # =========================================================== GUT Strategy Selector =======================================
    strategyProbablity1 = nashlh.lhmne(situationMatrixLevel1)
    strategyProbablity2 = nashlh.lhmne(situationMatrixLevel2)

    e1 = {}
    e21 = {}
    e22 = {}
    e23 = {}
    e24 = {}

    a1 = {}
    a21 = {}
    a22 = {}
    a23 = {}
    a24 = {}

    explorer_gut_cpt = {}
    alien_gut_cpt = {}
    first_level_strategy = {}
    second_level_strategy = {}

    explorer_strategy = []
    alien_strategy = []

    # build mixed nash equilibrium (MNE) table
    # first level
    e1, a1 = first_level_mne(e1, a1, strategyProbablity1)

    # second level
    e21, a21 = second_level_mne(e21, a21, strategyProbablity2)
    e22, a22 = second_level_mne(e22, a22, strategyProbablity2)
    e23, a23 = second_level_mne(e23, a23, strategyProbablity2)
    e24, a24 = second_level_mne(e24, a24, strategyProbablity2)

    # calculate explorers' different strategies combination conditional probability
    explorer_gut_cpt['e11a11e21t'] = GUT_CPT(e1['attacking'], a1['attacking'], e21, 'change_speed')
    explorer_gut_cpt['e11a11e21l'] = GUT_CPT(e1['attacking'], a1['attacking'], e21, 'change_direction')
    explorer_gut_cpt['e11a12e22t'] = GUT_CPT(e1['attacking'], a1['defending'], e22, 'change_speed')
    explorer_gut_cpt['e11a12e22l'] = GUT_CPT(e1['attacking'], a1['defending'], e22, 'change_direction')

    explorer_gut_cpt['e12a11e23t'] = GUT_CPT(e1['defending'], a1['attacking'], e23, 'change_speed')
    explorer_gut_cpt['e12a11e23l'] = GUT_CPT(e1['defending'], a1['attacking'], e23, 'change_direction')
    explorer_gut_cpt['e12a12e24t'] = GUT_CPT(e1['defending'], a1['defending'], e24, 'change_speed')
    explorer_gut_cpt['e12a12e24l'] = GUT_CPT(e1['defending'], a1['defending'], e24, 'change_direction')

    # calculate aliens' different strategies combination conditional probability
    alien_gut_cpt['e11a11a21t'] = GUT_CPT(e1['attacking'], a1['attacking'], a21, 'follow')
    alien_gut_cpt['e11a11a21t'] = GUT_CPT(e1['attacking'], a1['attacking'], a21, 'back')
    alien_gut_cpt['e11a12a22t'] = GUT_CPT(e1['attacking'], a1['defending'], a21, 'follow')
    alien_gut_cpt['e11a12a22t'] = GUT_CPT(e1['attacking'], a1['defending'], a21, 'back')

    alien_gut_cpt['e12a11a23t'] = GUT_CPT(e1['defending'], a1['attacking'], a23, 'follow')
    alien_gut_cpt['e12a11a23t'] = GUT_CPT(e1['defending'], a1['attacking'], a23, 'back')
    alien_gut_cpt['e12a12a24t'] = GUT_CPT(e1['defending'], a1['defending'], a24, 'follow')
    alien_gut_cpt['e12a12a24t'] = GUT_CPT(e1['defending'], a1['defending'], a24, 'back')

    explorer_first_level_strategy, explorer_second_level_strategy = final_explorer_strategy_combination(explorer_gut_cpt)
    alien_first_level_strategy, alien_second_level_strategy = final_alien_strategy_combination(alien_gut_cpt)
    # final_strategy_combination(gut_cpt)

    explorer_strategy.append(explorer_first_level_strategy)
    explorer_strategy.append(explorer_second_level_strategy)

    alien_strategy.append(alien_first_level_strategy)
    alien_strategy.append(alien_second_level_strategy)

    print("explorers' strategies are " + explorer_first_level_strategy + " and " + explorer_second_level_strategy)
    print("aliens' strategies are " + alien_first_level_strategy + " and " + alien_second_level_strategy)

    return(explorer_strategy, alien_strategy)