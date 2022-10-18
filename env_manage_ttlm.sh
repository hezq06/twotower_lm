conda env remove --name ttlm
conda create --name ttlm python=3.7 -y
source activate ttlm

conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
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

