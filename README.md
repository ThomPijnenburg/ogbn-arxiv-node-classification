# ogbn-arxiv-node-classification
Study of node classification approaches with GNN and BERT

## The Graph
We are looking at the citation network between Computer Science papers from arxiv by OGB ([`ogbn-arxiv`](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)) [1]. Each node is a paper and directed edges represent a citation between two papers. Each paper additionally comes with a 128-dimensional feature vector that is obtained by averaging the word embeddings from the title and abstract. These are precomputed by OGB using a skip-gram model over the corpus of papers. However, they have made a tsv file available with the title+abstracts of the papers, so we could try other embedding approaches (e.g. BERT).

## The prediction task
The task is to predict the 40 subject areas of Arxiv CS papers, e.g. cs.AI, cs.LG etc. These are manually assigned by the authors and Arxiv moderators, but an automatic system for the categorisation will help address the increasing rate of yearly growth of scientific literature.

## Splitting
Dataset splitting is based on the publication year of the papers. In the training set we find papers published until 2017, validation contains papers from 2018 and in test 2019.



# References

- [1] Hu, Weihua, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. ‘Open Graph Benchmark: Datasets for Machine Learning on Graphs’. arXiv, 24 February 2021. http://arxiv.org/abs/2005.00687.
