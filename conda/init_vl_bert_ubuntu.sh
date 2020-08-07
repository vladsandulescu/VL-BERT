conda create -n vl-bert python=3.6 pip --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vl-bert

cd $HOME/playground/hmm/lib
# git clone https://github.com/jackroos/VL-BERT.git

# conda install gcc-5 -c psi4 --yes
conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch --yes

cd $HOME/playground/hmm/lib/VL-BERT
pip install Cython
pip install -r requirements.txt
pip install torchvision==0.3.0

sudo rm -r $HOME/playground/hmm/lib/VL-BERT/common/lib/roi_pooling/build/
sudo rm $HOME/playground/hmm/lib/VL-BERT/common/lib/roi_pooling/C_ROIPooling.cpython-36m-x86_64-linux-gnu.so 

cd $HOME/playground/hmm/lib/VL-BERT
sh ./scripts/init.sh

cd $HOME/playground/hmm/lib/VL-BERT
mkdir model
cd model
mkdir pretrained_model
cd pretrained_model
gdown -O vl-bert-base-e2e.model --id 1jjV1ARYMs37tOaBalhJmwq7LcWeMai96
gdown -O vl-bert-large-e2e.model --id 1YTHWWyP7Kq6zPySoEcTs3STaQdc5OJ7f
gdown -O vl-bert-base-prec.model --id 1YBFsyoWwz83VPzbimKymSBxE37gYtfgh
gdown -O vl-bert-large-prec.model --id 1REZLN7c3JCHVFoi_nEO-Nn6A4PTKIygG
gdown -O bert.zip --id 14VceZht89V5i54-_xWiw58Rosa5NDL2H
unzip bert.zip
gdown -O resnet101-pt-vgbua-0000.model --id 1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE

pip install ipykernel
# python -m ipykernel install --user --name vlbert_p36 --display-name "conda_vlbert_p36"
python -m ipykernel install --prefix=/home/ubuntu/anaconda3 --name vl-bert --display-name "Python (vl-bert)"

cd $HOME
echo "Done"
