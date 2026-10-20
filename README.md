
# Satellite Image Segmentation

This project focuses on satellite image segmentation using PyTorch and a U-Net model. The goal is to identify and classify different land cover types in satellite images of Dubai.

## Dataset

We used the [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets) dataset.

Download the dataset and place it inside the `data/` directory in a folder named `dubai_datest`.

## Technologies Used

* Python
* PyTorch
* U-Net
* NumPy
* Jupyter Notebook

## Project Structure

* `code/` – Contains the implementation of the model, dataset handling, utilities, and training.
* `data/` – Contains the satellite image dataset.
* `notebooks/` – Contains notebooks for visualizing training metrics.
* `docs/` – Contains the project report.
* `requirements.txt` – Lists the required Python packages.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/inf2021090/AI_lab_Project.git](https://github.com/inf2021090/AI_lab_Project.git)
   cd AI_lab_Project

```

2. Install the required packages:
```bash
pip install -r requirements.txt

```


3. Download the dataset and place it in the `data/dubai_datest/` directory.
4. Navigate to the code directory and run the training script:
```bash
cd code
python train.py

```



## Results

Training metrics are saved in a CSV file and can be visualized using the Jupyter notebooks in the `notebooks/` directory.



Team
* Konstantinos Kafteranis 
* Jon Dorronsoro (2168329) 
