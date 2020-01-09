from ninolearn.learn.fit import cross_training
from mlr import mlr, pipeline, pipeline_noise

if __name__=="__main__":
    cross_training(mlr, pipeline_noise, 50, alpha=[0.,0.001],  name='mlr_review_noise')



