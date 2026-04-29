# Setup environment for running the DPG benchmark.

conda create -n dpg python=3.10 -y
conda activate dpg
pip install pip==23.3.2
pip install -r requirements.txt

# Some systems may require exporting a torch CUDA runtime path. Uncomment and
# adjust the value below to point at your own conda environment if needed:
# export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
