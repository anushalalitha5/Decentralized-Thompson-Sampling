# Decentralized-Thompson-Sampling

This repository contains code for:
- Experiments comparing decentralized Thompson Sampling algorithm's (proposed in this [paper](https://arxiv.org/pdf/2010.10569.pdf)) performance with prior work. 
- Experiments studying the effect of the number of agents and network topology on per-agent cumulative regret. 
- Experiments to simulate decentralized Thompson Sampling algorithm under gossip protocol and over time-varying networks, where each communication link has a fixed probability of failing.
- Experiments for general bandit problems where posterior cannot be computed in closed form, hence an approximate posterior in the class of Gaussian Mixture Models (GMM) is obtained using Variational Bayes Inference technique. This portion code has been CPU parallelized and gives gains of 12x.
