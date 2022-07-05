# Predicting Task-relevant Objects in Egocentric Videos

This is the implementation of EgoMotionNet.

## Environment
The models were implemented with Python 3.8.10 and Pytorch 1.10.0 on Ubuntu 20.04. For installing
other packages, please run `pip install -r requirements.txt`.



## Dataset
We use the Short-term Human-object Interaction Anticipation benchmark of
<a href="https://ego4d-data.org/#download">EGO4D</a> dataset. Please follow
this <a href="https://ego4d-data.org/docs/start-here/"> guideline </a> 
for accessing it. In our implementation, all frames are resized to 456x256.


## Train
``python main.py --data_path **/**/dataset/EGO4D/ --exp_path ***/***/experiments
--exp_name ssl --lr 3e-3 -bs 32 epoch 1000``
<br>
explanations:
<ul>
<li>data_path: path to folder of dataset</li>
<li>exp_path: path to folder of experiments</li>
<li>exp_name: specific name for the current experiment</li>
<li>lr: learning rate</li>
<li>bs: batch size</li>
<li>epoch: number of training epochs</li>
</ul>


## Evaluation
``python eval.py --data_path **/**/dataset/EGO4D/ --exp_path ***/***/experiments
--exp_name ssl  --model_path **/**/model_epoch_xx.pth``
<br>
explanations:
<ul>
<li>data_path: path to folder of dataset</li>
<li>exp_path: path to folder of experiments</li>
<li>exp_name: specific name for the current experiment</li>
<li>model_path: path to pre-trained model</li>
</ul>

Pretrained models are available in `./pre-trained_models/`

## Retrieval
``python retrieval.py --data_path **/**/dataset/EGO4D/ --exp_path ***/***/experiments
--exp_name retrieval   --model_path **/**/model_epoch_xx.pth``
<br>
explanations:
<ul>
<li>data_path: path to folder of dataset</li>
<li>exp_path: path to folder of experiments</li>
<li>exp_name: specific name for the current experiment</li>
<li>model_path: path to pre-trained model</li>
</ul>

Pretrained models are available in `./pre-trained_models/`

## Reference

This implementation is based on <a href='https://github.com/TengdaHan/DPC'> the implementation of DPC </a>