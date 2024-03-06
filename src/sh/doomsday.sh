## Remove old environments & kernels
jupyter kernelspec uninstall mistral_env
jupyter kernelspec uninstall eval_env
conda remove --name mistral_env --all
conda remove --name eval_env --all

## Create environment from .yml file
conda env create -f ./yml/mistral.yml
conda env create -f ./yml/env.yml

## Install ipykernel for further usage
python -m ipykernel install --user --name mistral_env --display-name "Patrick (Mistral)"
python -m ipykernel install --user --name eval_env --display-name "Patrick (Eval)"

## Download environment's model by running script
cd scripts

conda activate mistral_env
python mistral.py
conda deactivate

conda activate eval_env
python eval.py
conda deactivate

cd ..