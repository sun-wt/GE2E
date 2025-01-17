

### Environment

```bash
python=3.7
conda create --name GE2E python=3.7
conda install -c "nvidia/label/cuda-11.6.0" cuda-nvcc
conda install -c conda-forge cudnn=8.2.1.32
pip install -r requirements.txt
cd /datas/store162/syt/miniconda3/envs/GE2E/lib
ln -s libcusolver.so.11 libcusolver.so.10
# export export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/datas/store162/syt/miniconda3/envs/GE2E/lib
git clone https://github.com/sooftware/conformer.git
```

1. Generate the training dataset
```bash
cd generate

# change the mswc path in generate_mswc_pkl_above10.py
python generate_mswc_pkl_above10.py

# change the input_pkl path in generate_fixed_pkl.sh
bash generate_fixed_pkl.sh
```
2. Train the model
```bash
cd ..

# change the parameter in train.sh
bash train.sh
```

3. Inference
```bash
# change the checkpoint path in test.sh
bash test.sh
```
