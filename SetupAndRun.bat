python -m venv chessEnv
call chessEnv/Scripts/activate
python -m pip install --upgrade pip
pip install pygame==1.9.6
pip install librosa==0.8.0
pip install sounddevice==0.3.15
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-2.0.0-cp36-cp36m-win_amd64.whl
pip install keras==2.2.4 --force-reinstall
pip install numpy==1.16.0
python "source/Chess.py"