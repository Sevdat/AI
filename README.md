# AI
python -m venv .venv

activate

pip install numpy

pip install open3d

pip install pybind11-stubgen

pybind11-stubgen open3d -o typings

![AI Dataset](https://github.com/user-attachments/assets/72b967a5-24a4-46f9-8e83-610ad8a1014b)


generate clibrary code to be used in python
gcc -fPIC -shared -o clibrary.so clibrary.c