---
title: Adversarial Learning on Graph
description: This review gives an introduction to Adversarial Machine Learning on graph structured data, including several recent papers and research ideas in this field. 
author: Jintang Li
---

# Introduction

Graph Neural Networks (GNNs) has received considerable amount attention in recent years, which has achieve excellent performance in many graph analysis tasks such node classification, link prediction and graph clustering. However, more and more researches have pointed out the *vulnerability* of GNNs, that is, they are easily affected by several will-designed inputs. As shown in Fig.1, slightly modifying the existence of either edges or nodes may lead to the wrong prediction of the target node. Such unreliable results provide the opportunity for attackers to exploit these vulnerabilities and restrict the applications of GNNs.

<p align="center"> <img  width = "500"  height =300 src="https://github.com/gitgiter/Graph-Adversarial-Learning/blob/master/imgs/attack_demo.png" />

To resist such attacks, multiple methods have been explored in the literature including adversarial training, transfer learning, employing Gaussian distributions to represent nodes, recovering potential adversarial perturbations, allocating reliable queries, and certifiable robustness. Accordingly, there are a line of studies proposed to further exploit the vulnerabilities of GNNs, leading to the arms race in graph adversarial
learning.

In this review, we begin by providing an overview of adversarial learning on graphs, followed by several new challenges and open problems.

# Graph Adversarial Learning
## Adversarial Attack on Graph
First, we begin by providing some preliminary knowledge on how attacker works on graph data. According to [][], attack on graph can be categorized in to different types based on from different perspectives. Here we will briefly introduce some of the main categories.

+ **Poisoning and Evasion Attack**. Poisoning attack (a.k.a training-time attacks) means an attack occurred in the training stage, which tries to affect the performance of the targeted system (e.g., a GNN) by adding adversarial samples into the training dataset. In contrast, Evasion attack (a.k.a test-time attacks) means the target system is trained on a clean graph, and tested on a perturbed one.

+ **Targeted and Untargeted Attack**. The adversarial goal of Untargeted attack is to destroy the performance of GNNs on most of the instances, while the targeted attack aims to reduce the performance of some target instances.

+ **Black-, Grey-, and White-box Attack**. This type of attack is characterized bbased on different level of knowledge that attackers accessed. The information of attackers would receive is increasing from black-box attack to white-box attack.

Unlike, e.g., images consisting of continuous features, the graph structure – and often also the nodes’ features –- is discrete. It is difficult to design efficient and effective algorithms to cope with the underlying discrete domain. The first work of adversarial attack on graph data is proposed by Zügner et al.[]. An efficient algorithm named *Nettack* was developed based on a linear GCNs[]. Specifically, the output of a GCN with one hidden layer can be formulated as follows:
$$
Z=f_{\theta}(A, X)=\operatorname{softmax}\left(\hat{A} \sigma\left(\hat{A} X W^{(1)}\right) W^{(2)}\right)
$$
where $A$ is adjacency matrix, $\hat{A}$ is normalized laplacian matrix, $X$ is node feature matrix, $\sigma$ is activation function and $W$ is trainable weights of GCN.

*Nettack* drops the non-linear activation (e.g., ReLU) in hidden layer, so as to efficiently compute the perturbed output of $\hat{Z}$ as one edge or one node's features has changed. *Nettack* computes all possible changes of attack on a targeted node and chose the best perturbations within given budgets. It has been proved effective in attacking other GNNs except GCN, even under black-box attack setting.

In the following works, gradient-based methods have been widely used to find the optimal perturbations on the graph. Dai et al. adopt a surrogate model, such as GCN to compute the gradients of training loss w.r.t. the input graph. They choose the adversarial edge with the largest magnitude of gradients and flip it. Gradient-based methods are easy but effective in attacking GNNs, however, it takes much memory since computing the gradients of adjacency matrix is costly, approximating $O(N^2)$ where $N$ is the number of nodes in the graph.

To address this problem, this work [] proposed a *simplified gradient-based attack (SGA)*. As the main property of GNNs lies in aggregating messages from the node and its neighborhoods, attacking a targeted node could be simplified with its $k$-hop neighbors, where $k$ depends on the receptive fields of GNNs. The authors also uses a simplified GCN (namely SGC) as surrogate model, which can be formulated as follows
$$
Z=f_{\theta}(A, X)=\operatorname{softmax}\left(\hat{A}^2 X W\right)
$$

This is a two layer SGC and the computation of $\hat{A}^2 X$ could be done before training, which simplifies the training of GCN. Based on SGC and $two$-hop subgraph of targeted node, SGA achieves much more efficiency than *Nettack* and other gradient-based methods.

SGA only works for targeted attack, there lacks a more efficient algorithm for untargeted attack.

## Adversarial Defense on Graph
With graph data, recent intensive studies on adversarial attacks have also triggered the research on adversarial defenses. According to [][],  the defense methods are classified into three popular categories. 

+ **Adversarial Training**. While adversarial training has been widely used by attackers
to perform effective adversarial intrusion, the same sword can be used by defenders to improve the robustness of their models against adversarial attacks. That is, the model will be trained on a clean graph as well as perturbed graphs, where the perturbed graphs are generated by several representative attacks. In this way, the learned graph model is expected to be resistant to future adversarial attacks.

+ **Adversarial Detection**. Adversarial detection methods usually work as preprocessing methods on the graph. Instead of generating adversarial attacks during training, another effective way of defense is to detect and remove (or reduce the effect of) attacks, under the assumption that data have already been polluted.

+ **Robust Optimization**. These methods employ robust optimization strategy, which is not sensitive to extreme embeddings, to train the graph model. They often use more robust aggregation function of GNNs to improve the robustness.

Adversarial learning on graph data is first studied by Xu et al by solving a min-max problem:
$$
\min _{\theta} \max _{\hat{G}} \sum_{i} \mathcal{L}\left(f_{\theta}\left(v_{i}, \hat{G}\right), y_{i}\right)
$$
where $\hat{G}$ denotes the perturbed graph, $v_i$ denotes the node and $y_i$ is the corresponding class label. The perturbed graph is generated by a gradient-based attack algorithm, and it truly enhance the robustness of GNNs.

However, adversarial learning can only defense for evasion attacks, they are still strongly affected by poisoning attacks. To this end, adversarial detection methods have been proposed. Wu et al. observe that the adversarially manipulated graph differs from normal graphs statistically. For example, attackers tends to link dissimilar nodes and dislink similar nodes to destroy the message passing mechanism of GNNs. The authors propose to recovers the potential adversarial perturbation by Jaccard Similarity between nodes connected in the graph. Detection methods are simple but effective to  resist adversarial attack especially poisoning attacks. In the following works,  many works [][] adopt this strategy to *purify* the graph.

Rather than direct detection of suspicious nodes or edges before training, Zhu et al. designed a more robust GCN, which dynamically uncover and down-weigh suspicious data during training with attention mechanism.

## Robustness Certification on Graph 
Robustness of the graph model is always the main issue among all the existing works including attack and defense mentioned above. Therefore certificate the node’s absolute robustness under arbitrary perturbations is important.

# Conclusion
GNNs are vulnerable to adversarial attacks on the graph, we give an review of adversarial machine learning on graph, from the perspectives of attack, defense, robustness certification and stability.




