conda env remove --name mistral_env
conda env create -f ./conda/mistral.yml
python -m ipykernel install --user --name mistral_env --display-name "Patrick (Mistral)"