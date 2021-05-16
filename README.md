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
Each of the top level `*_inference.py` scripts map to a section in the thesis. They all leverage the helper-functions in [experiment_configs.py](experiment_configs.py).
