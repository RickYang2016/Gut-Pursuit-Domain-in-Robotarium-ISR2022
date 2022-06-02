# GUT for Multi-Robot Cooperative Pursuit Strategy

## Abstract
Underlying relationships among multiagent systems (MAS) in hazardous scenarios can be represented as game-theoretic models. In adversarial environments, the adversaries can be intentional or unintentional based on their needs and motivations. Agents will adopt suitable decision-making strategies to maximize their current needs and minimize their expected costs. This paper proposes and extends the new hierarchical network-based model, termed Game-theoretic Utility Tree (GUT), to arrive at a cooperative pursuit strategy to catch an evader in the Pursuit-Evasion game domain. We verify and demonstrate the performance of the proposed method using the [Robotarium platform](https://www.robotarium.gatech.edu/) compared to the conventional constant bearing (CB) and pure pursuit (PP) strategies. The experiments demonstrated the effectiveness of the GUT, and the performances validated that the GUT could effectively organize cooperation strategies, helping the group with fewer advantages achieve higher performance.

Relative paper: [Game-theoretic Utility Tree for Multi-Robot Cooperative Pursuit Strategy](https://github.com/RickYang2016/Gut-Pursuit-Domain-Robotarium-ISR2022/blob/main/Gut-Pursuit-Domain-Robotarium-ISR2022Paper.pdf)

## Pursuit-Evasion Game in Robotarium

<div align = center>
<img src="https://github.com/RickYang2016/Gut-Pursuit-Domain-Robotarium-ISR2022/blob/main/figures/GUT-PE-overview.png" height="200" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Gut-Pursuit-Domain-Robotarium-ISR2022/blob/main/figures/gut_pursuit_overview.png" height="200" alt="Hopper-V2 3SABC Video"/>
</div>

### Experiments Setup


### Demonstration: `Constant Bearing (CB)` vs `Pure Pursuit (PP)` vs `GUT`
> 1 Pursuer chasing 1 Evader 
    <div align = center>
    <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/cb.gif" height="133" title="Constant Bearing (CB)">   <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/pp.gif" height="133" alt="Hopper-V2 3SABC Video">      <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/gut_pursuit.gif" height="133" alt="Hopper-V2 3SABC Video"/>
    </div>
    
> 3 Pursuer chasing 1 Evader 
    <div align = center>
    <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/cb.gif" height="133" title="Constant Bearing (CB)">   <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/pp.gif" height="133" alt="Hopper-V2 3SABC Video">      <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/gut_pursuit.gif" height="133" alt="Hopper-V2 3SABC Video"/>
    </div>
    
> 5 Pursuer chasing 1 Evader 
    <div align = center>
    <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/cb.gif" height="133" title="Constant Bearing (CB)">   <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/pp.gif" height="133" alt="Hopper-V2 3SABC Video">      <img src="https://github.com/RickYang2016/PhD-Dissertation-SASS/blob/main/figures/gut_pursuit.gif" height="133" alt="Hopper-V2 3SABC Video"/>
    </div>

## Conclusion

Our work extends the Game-theoretic Utility Tree (GUT) in the pursuit domain to achieve multiagent cooperative decision-making in catching an evader. We demonstrate the GUT's performance in the real robot implementing the Robotarium platform compared to the conventional constant bearing (CB) and pure pursuit (PP) strategies. Through simulations and real-robot experiments, the results show that the GUT could effectively organize cooperation strategies, helping the group with fewer advantages achieve higher performance.

In our future work, we plan to improve GUT from different perspectives, such as optimizing GUT structure through learning from different scenarios, designing appropriate utility functions, building suitable predictive models, and estimating reasonable parameters fitting the specific scenario. Besides, optimizing GUT structure through learning from different scenarios with reinforcement learning techniques is also an avenue for future work. Especially, integrating deep reinforcement learning (DRL) into GUT will primarily increase its application areas and effectiveness.
