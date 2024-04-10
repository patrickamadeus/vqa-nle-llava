## Remove old environments & kernels
jupyter kernelspec uninstall main_env
conda remove --name main_env --all
# jupyter kernelspec uninstall eval_env
# conda remove --name eval_env --all

# ## Create environment from .yml file
conda env create -f ./yml/main.yml
# conda env create -f ./yml/eval.yml

## Install ipykernel for further usage
python -m ipykernel install --user --name main_env --display-name "Patrick (Main)"
# python -m ipykernel install --user --name eval_env --display-name "Patrick (Eval)"

## Download environment's model by running script
cd scripts

conda activate main_env
python main.py
conda deactivate

cd ..