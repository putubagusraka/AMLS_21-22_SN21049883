# AMLS_21-22_SN21049883

## Introduction

In this assignment, we have 2 tasks, those being to classify brain tumor MRIs into binary labels (task A) and multi-class labels (task B).

For task A, the binary classification is done using SVM and binary-label CNN. For task B, the multi-class classification is done using multi-class CNN.
Furthermore, after the training process, testing is done on the pure testing dataset released on the final week before the deadline, in which the trained model are tested alongside modified versions of the model for ablation study.

## Organization of Files
```bash
├───Ablation Assets
│   ├───Ablation_Task_A_CNN_Model
│   │   ├───assets
│   │   └───variables
│   └───Ablation_Task_B_CNN_Model
│       ├───assets
│       └───variables
├───Task A Assets
│   ├───.ipynb_checkpoints
│   └───Task_A_CNN_Model
│       ├───assets
│       └───variables
├───Task B Assets
│   ├───.ipynb_checkpoints
│   └───Task_B_CNN_Model
│       ├───assets
│       └───variables
├───dataset
│   ├───image (contains 3000 training images)
│   └───label.csv
├───test
│   ├───image (contains 200 testing images)
│   └───label.csv
├───Ablation_Task_A_CNN.ipynb
├───Ablation_Task_A_SVM.ipynb
├───Ablation_Task_B_CNN.ipynb
├───Training_Task_A_CNN.ipynb
├───Training_Task_A_SVM.ipynb
├───Training_Task_B_CNN.ipynb
├───Testing_All.ipynb
├───project_functions.py
├───ablation_functions.py
├───__init__.py
├───README.md
```
## File Roles
#### Base Directory
As mentioned above and as stated in the assignment prompt:

Task A uses 2 models: SVM and CNN

Task B uses 1 model: CNN

1) There are 7 main ipynb files (split due to computational restraints). Each can be separated into the 3 main processes of the assignment.

    a) For the main model training, we have `Training_Task_A_SVM.ipynb`, `Training_Task_A_CNN.ipynb`, and `Training_Task_B_CNN.ipynb`.
    
    b) For ablation studies (which are simply just slightly adjusted versions of the main training models), we have `Ablation_Task_A_SVM.ipynb`, `Ablation_Task_A_CNN.ipynb`, and `Ablation_Task_B_CNN.ipynb`. 

    c) For testing (all are combined into 1 file since there it is simply calling the already-trained models), we have the `Testing_All.ipynb`.

2) The .py files `project_functions.py` and  `ablation_functions.py` are packaged modules that are imported when running the `Ablation_*.ipynb`,
    `Training_*.ipynb`, `Testing_All.ipynb` scripts.

3) The file folders `dataset/` and `test/` contain the 3000 training images and 200 testing images respectively. In addition, each has a `label.csv` file that contains the label of each image.

4) There are 3 asset folders containing the trained models (used for testing or if user does not intend to rerun the training scripts), those being `Task A Assets/`, `Task B Assets/`, and `Ablation Assets/`.

## Code Execution

### Step-by-step

1) Clone/download git repository (https://github.com/putubagusraka/AMLS_21-22_SN21049883)
2) Download asset file from GDrive (https://drive.google.com/drive/folders/17FK40wSg1Or9Vr-PYvrjOeMALh05d2iS?usp=sharing)
3) Extract asset file as is into main directory, foldering already as needed. (should resemble the file directory tree stated above)

Based on desired usage, below are the steps on how to run.

#### For testing only
1) All the models are pretrained already, as seen from the zip extractions. Simply run the `Testing_All.ipynb` script.

#### Full run: Training and testing (both main and ablation)
1) Run `Training_Task_A_SVM.ipynb`, `Training_Task_A_CNN.ipynb`, and `Training_Task_B_CNN.ipynb` scripts.
2) Run `Ablation_Task_A_SVM.ipynb`, `Ablation_Task_A_CNN.ipynb`, and `Ablation_Task_B_CNN.ipynb` scripts. 
Each script in steps 1 and 2 will automatically overwrite previously saved models in `Task A Assets/`, `Task B Assets/`, and `Ablation Assets/` folders.
3) Run the `Testing_All.ipynb` script.

## Dependent Environment and Libraries

The whole project is developed in Python 3.8.8. Please note that using other Python versions may lead to unknown errors. Required libraries are shown below.
* imbalanced_learn==0.8.1
* imblearn==0.0
* joblib==1.1.0
* keras==2.6.0
* matplotlib==3.4.3
* numpy==1.21.3
* opencv_python==4.5.3.56
* pandas==1.3.4
* scikit_learn==1.0.1
* tensorflow==2.7.0
* tensorflow_gpu==2.6.0
