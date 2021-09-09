import customdatasets

# open napari instance for training dataset
# navigate with 'n' for next and 'b' for back on the keyboard
# you can do the same for the validation dataset
from visual import DatasetViewer
dataset_viewer_training = DatasetViewer(customdatasets.training_dataset)
dataset_viewer_training.napari()
