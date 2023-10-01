from generate_dataset import generate_dataset
from split_train_val import split_train_val
from remove_wrong_sized import remove_wrong
from model import allena_modello
from predict import generate_predictions
from merge_predictions import merge_predictions
from generate_output import generate_output

if __name__ == "__main__":
    generate_dataset()
    split_train_val()
    remove_wrong("train")
    remove_wrong("val")
    remove_wrong("test")
    allena_modello(1e-3, 5)
    generate_predictions()
    merge_predictions()
    generate_output()