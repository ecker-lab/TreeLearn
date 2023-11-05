
ENV_NAME='tree_learn_deimos'
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge timm==0.6.12 -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y 
conda install -c pyg pytorch-scatter -y
pip install -r requirements.txt


# build
cd ..
pip install -e .
pip install jupyter

# further packages
conda deactivate
