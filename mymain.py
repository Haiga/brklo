import hydra
import os
from omegaconf import OmegaConf

from source.helper.RerankerEvalHelper import RerankerEvalHelper
from source.helper.RerankerFitHelper import RerankerFitHelper
from source.helper.RerankerPredictHelper import RerankerPredictHelper
from source.helper.RetrieverEvalHelper import RetrieverEvalHelper
from source.helper.RetrieverFitHelper import RetrieverFitHelper
from source.helper.RetrieverPredictHelper import RetrieverPredictHelper


def fit(params):
    if params.model.type == "retriever":
        helper = RetrieverFitHelper(params)
        helper.perform_fit()
    elif params.model.type == "reranker":
        helper = RerankerFitHelper(params)
        helper.perform_fit()


def predict(params):
    if params.model.type == "retriever":
        helper = RetrieverPredictHelper(params)
        helper.perform_predict()
    elif params.model.type == "reranker":
        helper = RerankerPredictHelper(params)
        helper.perform_predict()


def eval(params):
    if params.model.type == "retriever":
        helper = RetrieverEvalHelper(params)
        helper.perform_eval()
    elif params.model.type == "reranker":
        helper = RerankerEvalHelper(params)
        helper.perform_eval()



def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
