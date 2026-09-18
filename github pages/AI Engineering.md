---
layout: default
title: AI Engineering
---
# TL;DR

Production AI engineering is an architecture-design problem. This article presents a framework for decomposing a product feature ($F$) into operational tasks:

$$  
F \longrightarrow T={t_1,\ldots,t_n},  
$$

and assigning those tasks to estimators such as language models, classifiers, retrieval systems, heuristics, or deterministic code.

Estimators are composed into an architecture:

$$  
A=(P,E,G,V,R,\Phi),  
$$

which defines task grouping, estimator assignment, execution dependencies, validation, retries, and fallbacks.

Architectures should be compared using total expected cost:

$$  
\mathbb E  
\left[  
\mathcal C_{\mathrm{direct}}(A)+\Lambda(A)  
\right],  
$$

where $\mathcal C_{\mathrm{direct}}$ includes execution and recovery costs, while $\Lambda$ captures the consequences of invalid, delayed, incomplete, or harmful results.