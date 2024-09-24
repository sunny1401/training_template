# Training Template
- This repository provides a template for optimized training of deep learning models with reproducible results. 
- It provides configurable code for training using pytorch-lightning for both Slurm and Single GPU.
- The repository has code for:
 - PyTorch Lightning Trainer and Pipeline 
 - HPO Pipeline Trainer using Ray-Tune  
 - Wandb Logger 
 - LR Scheduler to be used along with Trainer; which includes [LARS LR Scheduler from MAE](https://github.com/facebookresearch/mae/blob/main/util/lars.py)

## Customization
To customize the current repo, template the repo to create a new repo. After the repo is created, the package folder needs to be updated if you want to customize and change. This needs to be done as follows:

- remove the poetry.lock
- in pyproject.toml, update the lines 1 -7 as needed
- update references training/pipeline/trainer.py and training/pipeline/training.py [TODO - Add Github actions] - This would be handled automatically if changed via refactor in vscode
- Add python packages as needed


```bash
mv training <new package name>
pip install poetry
poetry install
```