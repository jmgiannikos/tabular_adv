from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from classifier_models import XGB
from tabularbench.datasets import dataset_factory

DEFAULT_CONFIG = {
    "dataset_name": "url",
    "classifier_cls": XGB
}

def train_pipeline(model_class, dataset):
    model = model_class()
    x,y = dataset.get_xy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) #admittedly very small test set, but this is more to get a rough idea of performance
    
    trained_and_tuned_model = model.tune_hyperparameters(x_train, y_train)

    y_pred = trained_and_tuned_model.predict(x_test)
    score = f1_score(y_test, y_pred)
    print(score)

    return trained_and_tuned_model 

if __name__ == "__main__":
    config=DEFAULT_CONFIG
    dataset_name = config["dataset_name"]
    dataset = dataset_factory.get_dataset(dataset_name)
    classifier_model_class = config["classifier_cls"]
    model = train_pipeline(classifier_model_class, dataset)