

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
```