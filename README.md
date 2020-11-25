![alt text](https://github.com/DelaunayTriangulationLearner/DTLearn/blob/main/logo.png?raw=true)

DTLearn is a multi-processing Python Package for Delaunay Triangulation Learner and Its Ensembles. It is an end-to-end open source platform for machine learning. 

DTLearn was originally developed to conduct machine learning with Delaunay triangulations. 
The system is general enough to be applicable in a wide variety of other domains, as well.

## Install

To install the prerequisites for the current release, which includes support for multi-CPU linux server

```
$ pip install numpy
$ pip install scipy
$ pip install pathos
```


#### *Try your first DTLearn program*

```shell
$ python
```

```python
from DTL import *
import itertools
from pathos.multiprocessing import Pool
from contextlib import closing
from Bagging_DTL_Multithread import *

n = 10000
p = 10
X_train = np.random.uniform(size=[n, p])
Y_train = np.sum(X_train ** 2, axis=1)
X_test = np.random.normal(size=[n, p])
Y_test = np.sum(X_test ** 2, axis=1)

bdtl = Bagging_DTL()
bdtl.max_depth = 2
bdtl.fit(X_train, Y_train)
print bdtl.predict(X_train)
```


## License

[Apache License 2.0](LICENSE)
