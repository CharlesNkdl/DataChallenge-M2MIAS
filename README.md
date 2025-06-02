# DataChallenge-M2MIAS

Description

For this competition, we have created a dataset containing laboratory analyses for several thousand patients. Each matrix in the dataset corresponds to the biological history of a patient - that is to say, all of their biological results on a 12-month period.

Each of these matrices is associated with a label indicating if the patient will suffer from T2D the next year. The label is 0 if the patient is negative, and 1 if they are positive. If your algorithms are successful, it means they are able to predict the onset of T2D up to a year in advance! Physicians are not always able to make that prediction: this would be a breakthrough in the field of AI-powered medicine.

The dataset contains 3D matrices of shape (n_samples, n_timesteps, n_features) where :

    n_samples is the number of patient histories in the dataset (53652 for training, 26826 for evaluation)
    n_timesteps is the number of months in each matrix (set to 12, one row per month)
    n_features is the number of features available for prediction (77)

The names of each feature are also given with the data. Refer to the next section to see how to load the files!
Loading data

Data is given in .npz format, which is a serialization format for numpy arrays. To load the data, simply use:

import numpy as np

with np.load("[PATH_TO_FILE]/training_data.npz", allow_pickle=True) as f:
    data = f["data"]
    feature_names= f["feature_labels"]

Where data contains the matrices for each patient, and feature_names contains the names of the features (for example, age, sex, calcium, creatinine, etc.). The same applies to evaluation_data.

Labels (whether the patient is positive or not) are given in .csv format. To load them, you can use:

import pandas as pd

labels = pd.read_csv("[PATH_TO_FILE]/training_labels.csv")

Evaluation

Submissions are evaluated on F1 score between the predicted probability and the observed target.
Submission File

For each row in the evaluation set, you must predict a probability for the Label variable. The file should contain a header and have the following format:

Id,Label
0,0
1,0
2,1
etc.

Id simply corresponds to the number of the row in the evaluation dataset. There should be 26826 rows, so your Id's will range from 0 to 26825.

Label contains the label for each matrix contained in evaluation_data.npz, in the same order.

Upon submission, you will see your score, computed over 50% of the evaluation dataset. The other 50% is kept secret until the end of the competition, and will be used for the final rankings.

Tip: Since you do not have the labels for the evaluation data, it is recommended to create your own training and test split within training_data.npz, so that you can test your models locally without creating a submission.

## Prérequis

*   Python 3.8+
*   [uv](https://github.com/astral-sh/uv) (installateur de paquets et gestionnaire d'environnement Python)

## Installation

1.  **Cloner le dépôt (ou créer la structure manuellement) :**
    ```bash
    git clone <URL_DU_REPO_SI_EXISTANT>
    cd mon_analyse_simple
    ```
    Si vous ne clonez pas, assurez-vous d'avoir le fichier `requirements.txt`.

2.  **Installer `uv` (si ce n'est pas déjà fait) :**
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows (PowerShell)
    irm https://astral.sh/uv/install.ps1 | iex
    ```

3.  **Créer et activer l'environnement virtuel :**
    ```bash
    uv venv .venv  # Crée l'environnement
    ```
    Activer :
    *   macOS/Linux : `source .venv/bin/activate`
    *   Windows (CMD) : `.venv\Scripts\activate.bat`
    *   Windows (PowerShell) : `.venv\Scripts\Activate.ps1`

4.  **Installer les dépendances :**
    ```bash
    uv pip install -r requirements.txt
    ```

## Utilisation

1.  **Lancer JupyterLab :**
    Une fois l'environnement activé et les dépendances installées, vous pouvez lancer JupyterLab :
    ```bash
    jupyter lab
    ```
    Cela ouvrira JupyterLab dans votre navigateur web par défaut.

2.  **Structure de dossiers suggérée :**
    *   `data/`: Pour stocker vos fichiers de données (CSV, Excel, etc.). Pensez à ajouter les gros fichiers de données au `.gitignore`.
    *   `notebooks/`: Pour vos notebooks Jupyter (`.ipynb`) où vous effectuez l'analyse.
    *   `scripts/`: Pour d'éventuels scripts Python réutilisables.
    *   `output/` ou `results/`: Pour sauvegarder les graphiques, les résultats, les fichiers CSV générés.

## Exemple de flux de travail

1.  Placez votre fichier de données (par exemple, `donnees_brutes.csv`) dans le dossier `data/`.
2.  Ouvrez JupyterLab (`jupyter lab`).
3.  Créez un nouveau notebook dans le dossier `notebooks/` (par exemple, `analyse_exploratoire.ipynb`).
4.  Dans le notebook, importez les bibliothèques nécessaires :
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Configuration de style pour Seaborn (optionnel)
    sns.set_theme(style="whitegrid")
    ```
5.  Chargez vos données :
    ```python
    df = pd.read_csv('../data/donnees_brutes.csv') # Ajustez le chemin si besoin
    print(df.head())
    print(df.info())
    print(df.describe())
    ```
6.  Procédez à votre analyse, visualisations, etc.

## Pour mettre à jour `requirements.txt`

Si vous installez de nouveaux paquets pendant votre développement (par exemple, `uv pip install une_nouvelle_lib`), mettez à jour votre `requirements.txt` avec :
```bash
uv pip freeze > requirements.txt
