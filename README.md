# mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding

This repository is the official implementation of [mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding](https://ieeexplore.ieee.org/document/9089251). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Repository Structure
- mg2vec/dataset/:
    - sample_metagraph_stats: Input file containing metagraph statistics. Here we only include a small sample of the dblp graph, which contains metagraphs up to size 4 only to limit the size of the file. Each row represents a relationship between a metagraph and two nodes. The first two columns are nodes' id, the third column is metagraph's id (starting with m) and the last column is the frequency of the metagraph instances appearing with the two nodes (staring with f).
- mg2vec/:
    - mg2vec.py: Implementation of mg2vec model. If you want to use other dataset, remember to change the dataset name in line 24 and file name in line 69.
	- task.py: Example code for the downstream task of relationship classification and corresponding evaluation.
- mg2vec/raw graph/:
	- as.lg, dblp.lg and linkedin.lg: Raw graph structure of the dataset. To generate the metagraphs and corresponding instances for each dataset, see http://www.yfang.site/data-and-tools/grami-with-automorphism and http://www.yfang.site/data-and-tools/submatch for detailed information.

## Train

To train the model in the paper, run this command:

```train
python ./mg2vec.py
```

## Results

Our model achieves the following performance on :

### Relationship prediction of mg2vec on LinkedIn, AS and DBLP

|      Dataset       |    LinkedIn     |       AS       |      DBLP      |
| ------------------ |---------------- | -------------- | -------------- |
|     Accuracy       |     61.24%      |     88.92%     |     89.90%     |

### Relationship search of mg2vec on LinkedIn, AS and DBLP

|      Dataset       |    LinkedIn     |       AS       |      DBLP      |
| ------------------ |---------------- | -------------- | -------------- |
|        MAP         |     57.39%      |     86.22%     |     97.90%     |

## Citation
If you find this useful for your research, we would be appreciated if you cite the following paper:
```
@ARTICLE{9089251,
  author={W. {Zhang} and Y. {Fang} and Z. {Liu} and M. {Wu} and X. {Zhang}},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2020.2992500},
  ISSN={1558-2191},
  month={},}
```