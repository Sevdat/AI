# AI
python -m venv .venv

activate

pip install numpy

pip install oped3d

pip install pybind11-stubgen

pybind11-stubgen open3d -o typings
