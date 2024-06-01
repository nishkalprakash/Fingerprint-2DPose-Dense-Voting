- create a new environment
```
conda create -n pose python pytorch==1.10.0 opencv==3.4.2.17 numpy scipy imageio PyYaml Pillow
pip3 install torch torchvision  --index-url https://download.pytorch.org/whl/cu121
```
- delete the environment
conda remove -n pose --all