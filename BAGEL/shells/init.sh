# Install environment for BAGEL training.
#
# Step 1 (optional): create a conda environment.
#   conda create -n bagel python=3.10.6 -y
#   conda activate bagel
#
# Step 2: install Python dependencies.

pip install -r requirements.txt
pip install flash-attn==2.7.2.post1 --no-build-isolation
