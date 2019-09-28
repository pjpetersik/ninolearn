from ninolearn.learn.fit import cross_training
from mlr import mlr, pipeline

if __name__=="__main__":
    cross_training(mlr, pipeline, 50, alpha=[0.,0.001],  name='mlr')



