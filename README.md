# Experiments on deep neural networks for task-oriented conversational agents

This fork is based on the work *Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog* by Qin et al. 2020. Please refer to the original repository for information on that paper.
I used this codebase to carry out further experiments, some of which are meant to benchmark a different type of conversational agent (not neural network-based).

## Changes included here
Changes include:

* Modifications to generate and use a smaller version of the data sets considered in the paper (SMD aka KVR, and MultiWOZ 2.1), see `generate_our_test.py` and files referenced there.
* Minor changes to the original code base to test DF-Net on such data sets.
* Code to query GPT-3 on those data sets, using OpenAI's API. 