pip install wandb seaborn plotly
mkdir downloads
cd downloads
git clone https://github.com/pytorch/fairseq
cd fairseq
pip uninstall enum34
pip install --editable ./
cd ../..
pip install soundfile
apt update
apt install libsndfile1
