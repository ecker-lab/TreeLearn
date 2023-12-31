
ENV_NAME='TreeLearn'
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge timm==0.6.12 -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y 
pip install -r setup/requirements.txt


# build
pip install -e .
pip install jupyter
conda deactivate
