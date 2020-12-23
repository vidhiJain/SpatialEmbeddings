# Spatial Embeddings

Code for [Learning Embeddings that Capture Spatial Semantics for Indoor Navigation](/pubs/ORLR.pdf), presented at the Workshop on Object Representations for Learning and Reasoning at Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS) 2020.


## Introduction

Consider an example of finding a key-chain in a living room. A key-chain is found either on a coffee table, inside a drawer, or on a side-table. When tasked with finding the key-chain, a human would first scan the area coarsely and then navigate to likely locations where a key-chain could be found, for example, a coffee table. On getting closer, they would then examine the area (top of the table) closely. Following this, if the key-chain is not found, they would try to navigate to the next closest place where the key-chain could be found. The presence (or absence) of a co-located objects would boost (or dampen) their confidence in finding the object along the chosen trajectory. This would, in turn, influence the next step (action) that they would take. 

Our goal is to enable embodied AI agents to navigate based on such object-based spatial semantic awareness. To do this, we focus on the following problems: a) training object embeddings that semantically represent spatial proximity, and b) utilizing these embeddings for semantic search and navigation tasks. 


## Setup
1. Installations 

* Clone this repository 
```
git clone https://github.com/vidhiJain/SpatialEmbeddings.git
cd SpatialEmbeddings
```

* Install the dependencies

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:
```
conda env create -f conda_env.yml
conda activate spatialEmbeddings
```

* Download `word2vec` pre-trained on Google News corpus [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) into `SpatialEmbeddings/data/`. Remember to unzip it. 


2. Running the experiment 

```
python3 run.py
```
This creates the `results_<room_type>_queries.csv` for the floor-type specified. This contains Embedding name, Scene, Query object, Seed used to randomize the scene, and output value. The output value is as defined:

* `{sp, p}` : `p` Path length taken by the robot to reach the query object, vis-a-vis `sp` shortest path (if object location was known a priori.)
* `-1`: object not present in the scene.
* `-2`: no (unvisited) object in view to navigate to as a sub-goal.
* `-3`: robot oscillations exceed the threshold.

To modify parameters, including floor-type, refer `run.py`.

## Embeddings: 

File `embeddings.py` has the API interface for incorporating different object embeddings: Word2Vec, FastText, RoboCSE, and Graph-based embeddings. 

## Cite
```
@inproceedings{jain2020,
  title = {Learning Embeddings that Capture Spatial Semantics for Indoor Navigation},
  author = {Jain, Vidhi and Patil, Shishir and Agarwal, Prakhar and Sycara, Katia},
  booktitle = {NeurIPS 2020, Workshop on Object Representations for Learning and Reasoning},
  year = {2020}
}
```

## Debugging

For issues related to Ai2Thor, please refer to their [Ai2Thor repo](https://github.com/allenai/ai2thor). Please [raise an issue](https://github.com/vidhiJain/SpatialEmbeddings/issues), or write to us if you face any other issues. 

## Contact Us

We welcome comments, and criticism. For questions, please write to vidhij@andrew.cmu.edu. 

