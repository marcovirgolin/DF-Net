# Experiments on deep neural networks for task-oriented conversational agents

This fork is based on the work *Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog* by Qin et al. 2020. Please refer to the original repository for information on that paper.
I used this codebase to carry out further experiments for our paper:

```
@article{WahdeVirgolin2022DAISY,
  author = {Mattias Wahde and Marco Virgolin},
  title = {DAISY: An Implementation of Five Core Principles for Transparent and Accountable Conversational AI},
  journal = {International Journal of Human–Computer Interaction},
  volume = {0},
  number = {0},
  pages = {1-18},
  year  = {2022},
  publisher = {Taylor & Francis},
  doi = {10.1080/10447318.2022.2081762},
  URL = {https://doi.org/10.1080/10447318.2022.2081762},
  eprint = {https://doi.org/10.1080/10447318.2022.2081762}
}
```

## Changes included here
Changes include:

* Modifications to generate and use a smaller version of the data sets considered in the paper (SMD aka KVR, and MultiWOZ 2.1), see `generate_our_test.py` and files referenced there.
* Minor changes to the original code base to test DF-Net on such data sets.
* Code to query GPT-3 on those data sets, using OpenAI's API. 
