Learning Continuous Semantic Representations of Symbolic Expressions
===
This is the code relating to the paper [link](https://arxiv.org/abs/1611.01423).
More information, visualization and data related to this work can be found at the project [website](http://groups.inf.ed.ac.uk/cup/semvec/).
```
@article{allamanis2016learning,
         title={Learning Continuous Semantic Representations of Symbolic Expressions},
         author={Allamanis, Miltiadis and Chanthirasegaran, Pankajan and Kohli, Pushmeet and Sutton, Charles},
         journal={arXiv preprint arXiv:1611.01423},
         year={2016}
}
```

The code is written in Python 3.5 using Theano.

To train an eqnet run
```
python encoders/rnn/trainsupervised.py <trainingFileName> <validationFileName>
```
a `.pkl` file will be produced containing the trained network.