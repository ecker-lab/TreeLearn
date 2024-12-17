
ENV_NAME='TreeLearn'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# conda
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install mkl=2024.0.0 -y # this might be required to fix some bug in this pytorch version (remove if not needed)
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y # this might be required to fix some cudnn laoding error that might occur on some systems (remove if not needed)
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y 

# additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
pip install -r setup/requirements.txt

# build
pip install -e .
conda deactivate
