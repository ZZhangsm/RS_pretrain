# Recommendation with GNN and embedding pretrain

The pipeline is based on the RS-Zoos.

We adapted LightGCN to utilise Bert Embedding for textual data（review in the Yelp）, and also initialised the initial embedding of LightGCN  using textual Bert Embedding. A significant improvement in experimental results is shown.

 If you want to turn on the switch for pre-training embedding, set '-pretrain' to 1. 

Of course, before doing so please make sure that you have stored user/item embedding in the appropriate directory.

--------------------------------------------------------------------------------------

## RS-Zoos

A pipeline with some recommender system models.

Basically can achieve the performance of the original paper report. 

Just for learning, many places need to be modified by yourself, such as using Adversarial Training on NGCF. 

Support three loss functions：BPR Loss, BCE Loss, and APR loss.

## Available models

| Model    | Original paper                                               | Publication |
| -------- | ------------------------------------------------------------ | ----------- |
| BPRMF    | BPR: Bayesian Personalized Ranking from Implicit Feedback    | UAI'09      |
| NCF      | Neural Collaborative Filtering                               | WWW'17      |
| NGCF     | Neural Graph Collaborative Filtering                         | SIGIR'19    |
| LightGCN | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | SIGIR'20    |
| AMF      | Adversarial Personalized Ranking for Recommendation          | SIGIR'18    |
| ALGN     | Apply APR on LightGCN                                        |             |
| LGNGuard | Apply "GNNGuard: Defending Graph Neural Networks against Adversarial Attacks" on LightGCN | NIPS'20     |

