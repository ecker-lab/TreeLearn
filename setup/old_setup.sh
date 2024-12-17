# setup with older python, pytorch and cuda version (also compatible with the repository)
ENV_NAME='TreeLearn'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

# conda
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install mkl=2024.0.0 -y # needs to downgraded because of some bug in other version
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y 

# additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
pip install -r setup/requirements.txt

# build
pip install -e .
conda deactivate
