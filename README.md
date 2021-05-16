# Proactive Policy Inference
This repository builds on the asignments from a _Formal Methods in Robotics_ course at WPI which evolved into my Master's thesis, [hosted by WPI](https://web.wpi.edu/Pubs/ETD/Available/etd-052818-100711/unrestricted/poulin_proactive_learning.pdf).

## Abstract
In multi-agent Markov Decision Processes, a controllable agent must perform optimal planning in a dynamic and uncertain environment that includes another unknown and uncontrollable agent. Given a task specification for the controllable
agent, its ability to complete the task can be impeded by an inaccurate model of
the intent and behaviors of other agents. In this work, we introduce an active policy
inference algorithm that allows a controllable agent to infer a policy of the environmental agent through interaction. Active policy inference is data-efficient and
is particularly useful when data are time-consuming or costly to obtain. The controllable agent synthesizes an exploration-exploitation policy that incorporates the
knowledge learned about the environment’s behavior. Whenever possible, the agent
also tries to elicit behavior from the other agent to improve the accuracy of the
environmental model. This is done by mapping the uncertainty in the environmental model to a bonus reward, which helps elicit the most informative exploration,
and allows the controllable agent to return to its main task as fast as possible. Experiments demonstrate the improved sample efficiency of active learning and the
convergence of the policy for the controllable agents.

## Navigating this repo
There are three main stages executed in the algorithm in my thesis.
 - Define a Markov Decision Process that represents a task to be completed by an agent.
 - Develop a novel policy inference algorithm to infer policies of observed agents.
 - Train a controllable agent to complete its task better, by giving it a bonus reward for learning the policy of another agent, that may be colaborative, antagonistic, or just following a random policy.
 
### Defining an agent that can solve a Markov diciesion process
 We specify an agents task (e.g., move through a grid world and "reach the green square while avoiding the red obstacles") with [Linear temporal logic (LTL)](https://github.com/nolanpster/proactive_policy_inference/blob/master/two_stage_proactive_inference.py#L44-L48) and convert it to a [deterministic raban automata (DRA)](https://github.com/nolanpster/proactive_policy_inference/blob/master/experiment_configs.py#L88) to form a state graph. The product of a DRA and a [Markov Decision Process (MDP)](https://github.com/nolanpster/proactive_policy_inference/blob/master/experiment_configs.py#L119) can be solved with the common [value iteration](https://github.com/nolanpster/proactive_policy_inference/blob/master/MDP_EM/MDP_EM/MDP_solvers.py#L40), and more quickly with [expectaiont maximization](https://github.com/nolanpster/proactive_policy_inference/blob/master/MDP_EM/MDP_EM/MDP_solvers.py#L103).

### Policy Inference with geodescic gaussian kernels
To infer a policy, we assume that a policy can be defined by a mixture of geodesic gaussian kernels, each with a contributing weight. These weights make up a paramater vector to be [inferred](https://github.com/nolanpster/proactive_policy_inference/blob/master/single_agent_inference.py#L205-L236) by observing the agent over sets of state transitions, or "rollouts".

### Proactive Inference with multiple agents
The culminating analysis script, [two_stage_proactive_inference.py](https://github.com/nolanpster/proactive_policy_inference/blob/9de478edea1f37b4e4910f6df5111b15415f8626/two_stage_proactive_inference.py), has two agents, one who is "controllable" and another that follows an unknown control policy. The controllable agent is given the LTL task listed above, "get to green, avoid red", and define that it also fails to complete its task if it "collides" with (a.k.a occupies the same grid-cell as) the uncontrollable agent. After each trajectory where the controllable agent attempts to get to the green cell, it infers the policy of the uncontrollable agent. To encourage learning about unknown parts of the uncontrollable agent's policy, I added a ["bonus reward"](https://github.com/nolanpster/proactive_policy_inference/blob/9de478edea1f37b4e4910f6df5111b15415f8626/experiment_configs.py#L302-L305) for minimizing the uncertainty of the parameters are assumed to make up the uncontrollable agent's policy.

## More details
Each of the top level `*_inference.py` scripts map to a section in the thesis. They all leverage the helper-functions in [experiment_configs.py](experiment_configs.py). A presentation that goes into more details is [here](https://github.com/nolanpster/proactive_policy_inference/files/6490163/Nolan_Poulin_Masters_thesis_presentation.pdf), though several slides have PowerPoint animations that have not been optimized for a PDF.
