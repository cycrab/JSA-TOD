# JSA-TOD on MultiWOZ2.1
This is the code and data for MultiWOZ2.1 experiments.
## Requirements
After you create an environment with `python 3.6`, the following commands are recommended to install the corresponding package.
* pip install torch==1.7.1
* pip install transformers==4.8.2
* pip install spacy==3.1
* python -m spacy download en_core_web_sm
* pip install nltk
* pip install sklearn
* pip install tensorboard
* pip install future

## Data preprocessing
First, you need to extract data from zip files
```
cd data
unzip MultiWOZ_2.1.zip
```
```
unzip mwzeval.zip
```
Then preprocess these files
```
python data_analysis.py
python preprocess.py
```
## Training
Before training, you need to create a directory to store log files
```
mkdir log21
```
### Supervised training over all data in MultiWOZ2.1:
You can train the end-to-end model with all training data in MultiWOZ2.1 
```
bash shell_scripts/train.sh $GPU
```
You can change other parameters such as batch size and learning rate in this `.sh` file. For instance, if your GPU has enough memory, you can increase the batch size and decrease the gradient accumulation steps.

*Note: Due to OS differences, if there is a problem running the `.sh` file, please execute the following command first*

```
dos2unix ${sh_file_name}
```
### Supervised pre-training
Our semi-supervised learning is divided into two stages. The first stage is supervised pre-training over small-sized labeled data and the second stage is semi-supervised training over the mix of labeled and unlabeled data.

During the first stage, to train the generative model, run
```
bash shell_scripts/pretrain_turn.sh $GPU $ratio
```
`ratio` denotes the amount of labels in terms of the proportions with respect to the full set of data. For example, if `ratio=20`, then 20% of data in MultiWOZ are regarded as labeled data, and the rest are regarded as unlabeled data.

To train the inference model, run
```
bash shell_scripts/pretrain_post_turn.sh $GPU $ratio
```
### Semi-supervised training
After supervised pre-training of the generative model and inference model, you can run either the JSA learning method or variational learning methods.
```
bash shell_scripts/train_jsa_turn.sh $GPU $ratio
bash shell_scripts/train_vl_turn.sh $GPU $ratio
```

## Evaluation 
To test the performance of your model on the test set of MultiWOZ2.1:
```
bash test_turn.sh $GPU $path
```
## Checkpoint
Our models and results are stored, and we can provide the checkpoints and results if needed.  

Once you have these checkpoints to folder `experiments_21`, and change the `path` of `test_turn.sh`, then you can get our results in paper
