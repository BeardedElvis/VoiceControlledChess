VoiceControlledChess

Setup instructions:

This project was built for python 3.6.5 64-bit
To set up, install python 3.6.5 and make sure this version will be run when running the command "python" in the command prompt.
Then run the file SetupAndRun which will execute the following commands:

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

Any data used to train the models (by running the file "trainModels.py") should be placed in
VoiceControlledChess/input/audio/trainRecordedNumbers and VoiceControlledChess/input/audio/trainRecordedLetters
for numbers and letters respectively. The testing data should similarly be placed in
VoiceControlledChess/input/audio/testRecordedNumbers and VoiceControlledChess/input/audio/testRecordedLetters

Game instructions:
To choose a file, click the button "Choose file". You will then have two seconds to speak the name of a file using the
NATO phonetic alphabet: Alfa, Bravo, Charlie, Delta, Echo, Foxtrot, Golf or Hotel

To choose a rank, click the button "Choose rank". You will then have two seconds to speak the name of a rank:
One, Two, Three, Four, Five, Six, Seven or Eight

When the correct tile has been chosen, press enter to confirm