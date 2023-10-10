conda env remove --name ttlm
conda create --name ttlm python=3.8 -y
source activate ttlm

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda jupyter
conda install -y nb_conda_kernels ipykernel
conda install -y -c conda-forge jupyter_contrib_nbextensions
conda install -y -c anaconda scikit-learn
conda install -y pip tqdm
conda install -y -c conda-forge tensorboardx
conda install -y -c anaconda h5py
conda install -y -c conda-forge mpi4py
conda install -y -c conda-forge pytorch-pretrained-bert
pip install deepspeed
pip install transformers

