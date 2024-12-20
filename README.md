# AI60301 Final Project
20245360 Yeseong Jung

This repository contains scripts for experiments involving the OCTGAN Synthesizer based on [[arxiv]](https://arxiv.org/abs/2105.14969). These experiments were performed as part of the AI60301 final project.

## Requirements
Run the following command in your terminal to install all the necessary packages:
```
pip install -r requirements.txt
```

## Experiments
To execute the OCTGAN, simply run the following command in your terminal:
```
python main.py
```
Run the above command to start training the OCTGAN model, generating synthetic data, and evaluating its performance.
 - Settings:  
    - Dataset: loan  [[Big Data Finance]](https://bigdata-finance.kr/dataset/datasetView.do
    )
    - Epochs: 100
    - Batch Size: 500
    - Learning Rate: 2e-3
    - If needed, you can customize the hyperparameters by editing the script or using additional arguments.
- Process Overview
    - Train OCTGAN
    - Generate synthetic data using the trained OCTGAN model.
    - Train machine learning models (AdaBoost, Decision Tree, and MLP) on the generated synthetic data.
    - Test the trained models on the test set to evaluate how well models trinaed on synthetic data generalize to real-world tasks.
    - Evaluation Metrics: Accurac, Binary F1 Score, etc.
- You can reproduce the results in the original paper by running python main.py --dataset_name 'adult'.

## Results
### loan (new dataset)
Below are the evaluation metrics for models trained on synthetic data generated by OCTGAN and tested on the real test set.  
| Metric                          | Value     |
|---------------------------------|-----------|
| Accuracy                        | 0.802     |
| Binary F1                       | 0.890     |
| Macro F1                        | 0.446     |
| Matthews Correlation Coefficient | -0.007    |
| Precision                       | 0.804     |
| Recall                          | 0.997     |
| ROC AUC                         | 0.346     |
| Silhouette Score                | 0.120     |

The binary F1 (0.890) and accuracy (0.802) show good performance on the binary classification task, indicating that the model trained on synthetic data generalizes well to real data. The macro F1 (0.446) and Matthews correlation coefficient (-0.007) suggest potential difficulties in handling imbalanced or multi-class data distributions. The ROC AUC (0.346) suggests limited discriminative ability between classes, most likely due to class imbalance. The silhouette score (0.120) indicates poor clustering quality on synthetic data, suggesting that improvements are needed to preserve the data structure.

Overall, the results highlight areas where OCTGAN is effective in generating useful synthetic data, but needs further optimization to handle complex distributions.

### adult (original dataset)
Below are the evaluation results for the 'adult' data. 

| Metric                      | Value |
|----------------------------------|-----------|
| Accuracy                        | 0.751     |
| Binary F1                       | 0.053     |
| Macro F1                        | 0.454     |
| Matthews Correlation Coefficient | 0.031     |
| Precision                       | 0.533     |
| Recall                          | 0.036     |
| ROC AUC                         | 0.416     |
| Silhouette Score                | 0.344     |
