Blindfolded Attackers Still Threatening: Strict Black-Box Adversarial Attacks on Graphs
===============================================================================

About
-----

This project is the implementation of the paper "Blindfolded Attackers Still Threatening: Strict Black-Box Adversarial Attacks on Graphs".
A strict black-box adversarial attack on graphs is proposed, where the attacker has no knowledge of the target model and no query access to the model. With the mere observation of the graph topology, the proposed attack strategy aim to flip a limited number of links to mislead the graph model.

This repo contains the codes, data and results reported in the paper.

Dependencies
-----

The script has been tested running under Python 3.7.7, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `scipy==1.4.1`
- `scikit-learn==0.23.1`
- `gensim==3.8.0`
- `networkx==2.3`
- `tqdm==4.46.1`
- `torch==1.4.1`
- `torch_geometric==1.5.0`
  - torch-spline-conv==1.2.0
  - torch-scatter==2.0.4
  - torch-sparse==0.6.0

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Usage: Node-level Attack
-----
Given the adjacency matrix of input graph, our attacker aims to flip a limited number of links. 

### Input Format
Following our settings, we only need the structure information of input graphs to perform our attacks.
An example data format is given in ```data``` where dataset is in ```npz``` format.

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).

### Output Format
The program outputs to a file in ```npz``` format which contains the adversarial edges.

### Main Script
The help information of the main script ```node_level_attack.py``` is listed as follows:

    python node_level_attack.py -h
    
    usage: node_level_attack.py [-h][--dataset] [--pert-rate] [--threshold] [--save-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be perturbed on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --threshold               float, Restart threshold of eigen-solutions.
      --save-dir                str, File directory to save outputs.
      
### Demo
We include all three benchmark datasets Cora-ML, Citeseer and Polblogs in the ```data``` directory.
Then a demo script is available by calling ```attack.py```, as the following:

    python attack.py --data-name cora --pert-rate 0.1 --threshold 0.03 
      
### Evaluations
Our evaluations depend on the output adversarial edges by the above attack model.
We provide the evaluation codes of our attack strategy on the node classification task here. 
We evaluate on three real-world datasets Cora-ML, Citeseer and Polblogs. 
Our setting is the poisoning attack, where the target models are retrained after perturbations.
We use [GCN](https://arxiv.org/pdf/1609.02907.pdf), [Node2vec](https://arxiv.org/pdf/1607.00653.pdf) and [Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) as the target models to attack.

#### Datasets
We evaluate on three real-world datasets Cora-ML, Citeseer and Polblogs. 
The preprocessed version is given in ```data``` directory where dataset is in ```npz``` format.

#### Evaluation Script
If you want to attack GCN, you can run ```evaluation/eval_gcn.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --dimensions              str, Dimensions of GCN hidden layer. Default is 16.
      --load-dir                str, File directory to load adversarial edges.
       
       
If you want to attack Label Propagation, you can run ```evaluation/eval_emb.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--window-size] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --dimensions              int, Output embedding dimensions of Node2vec. Default is 32.
      --window-size             int, Context size for optimization in Node2vec. Default is 5.
      --walk-length             int, Length of walk per source in Node2vec. Default is 80.
      --walk-num                int, Number of walks per source in Node2vec. Default is 10.
      --p                       float, Parameter in node2vec. Default is 4.0.
      --q                       float, Parameter in node2vec. Default is 1.0.
      --worker                  int, Number of parallel workers. Default is 10.
      --load-dir                str, File directory to load adversarial edges.
      
If you want to attack Node2vec, you can run ```evaluation/eval_lp.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --load-dir                str, File directory to load adversarial edges.

<!--
#### Performance
| Decrease in Macro-F1 score (%) |     GCN     |   Node2vec  | Label Prop. |
| :----------------------------: | :---------: | :---------: | :---------: |
|              Cora-ML           |     5.27    |     8.92    |    7.13     |
|              Citeseer          |     3.98    |     9.32    |    8.16     |
|              Polblogs          |     5.32    |     3.79    |    6.14     |
-->

 
Usage: Graph-level Attack
-----
Given a set of input graphs, our attacker aims to flip a limited number of links for each graph. 


### Input Format
When using your own dataset, you must provide:

* the adjacency matrix of a set of graphs.

### Main Script
The help information of the main script ```graph_level_attack.py``` is listed as follows:

    python graph_level_attack.py -h
    
    usage: graph_level_attack.py [-h][--dataset] [--pert-rate] [--threshold] [--model] [--epoch]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be perturbed on [ENZYMES, PROTEINS].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --threshold               float, Restart threshold of eigen-solutions.
      --target-model            str, The target model to be attacked on [gin, diffpool].
      --epochs                  int, The number of epochs.
      
### Demo
A demo script is available by calling ```graph_level_attack.py```, as the following:

    python graph_level_attack.py --data-name ENZYMES --pert-rate 0.2 --threshold 1e-5 --target-model diffpool --epochs 21
    
### Evaluations
For graph-level attack, we perform our attack strategy to the graph classification task. 
We use [GIN](https://arxiv.org/pdf/1810.00826.pdf) and [Diffpool](https://arxiv.org/pdf/1806.08804.pdf) as our target models to attack.
By running the script ```graph_level_attack.py```, you can directly get the evaluation results.

#### Datasets
We evaluate on two protein datasets: Enzymes and Proteins. 
We call ```torch_geometric``` package to download and load these two datasets.

<!--
#### Performance
| Decrease in Macro-F1 score (%) |     GIN     |   Diffpool  | 
| :----------------------------: | :---------: | :---------: | 
|              Proteins          |    13.53    |    24.88    |   
|              Enzymes           |    39.90    |    39.62    |   
-->
