jupyter kernelspec uninstall sampleenv
conda remove --name sampleenv --all

# conda create -n sampleenv -y python=3.9

# ## Install ipykernel for further usage
# python -m ipykernel install --user --name sampleenv --display-name "Patrick (Sample)"

## Download environment's model by running script
cd scripts

# conda activate sampleenv
python sample.py
# conda deactivate

cd ..