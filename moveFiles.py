
import os
import os.path

# test_data = pd.read_csv('./data/train.tsv', sep='\t')

#folder = 2

# while folder <= 30: 
# counter = 0
# for path in test_data['path']:
#     if not os.path.exists("D:/Documents/GitHub/VoiceControlledChess/data/clips/" + path[:-4] + ".mp3") and not os.path.exists("D:/Documents/GitHub/VoiceControlledChess/data/training/" + path):
#         print(path[:-4] + ".mp3")
#         if os.path.exists("D:/Documents/GitHub/VoiceControlledChess/data/all clips/" + path[:-4] + ".mp3"):
#             os.rename("D:/Documents/GitHub/VoiceControlledChess/data/all clips/" + path[:-4] + ".mp3", "D:/Documents/GitHub/VoiceControlledChess/data/mp3training/" + path[:-4] + ".mp3")
            # counter += 1
    # if counter >= 2000:
        # break
    #folder += 1

path = "CrewMemberWalk1_00000"
for i in range(251):
    if i == 10:
        path = "CrewMemberWalk1_0000"
    if i == 100:
        path = "CrewMemberWalk1_000"
    if os.path.exists("D:/Documents/GitHub/Dx12/assets/Walking/" + path + str(i) + ".obj"):
        os.rename("D:/Documents/GitHub/Dx12/assets/Walking/" + path + str(i) + ".obj", "D:/Documents/GitHub/Dx12/assets/Walking/" + str(i) + ".obj")

# import shutil

# src = 'C:/Users/Leo/Desktop/Photos from phone/Deleted pictures/Recovered data 04-20-2020 at 23_24_46/exFAT/DCIM/Niki'
# dest = 'H:/DCIM/Niki'

# src_files = os.listdir(src)
# for file_name in src_files:
#     full_file_name = os.path.join(src, file_name)
#     if os.path.isfile(full_file_name) and not os.path.exists(os.path.join(dest, file_name)):
#         shutil.copy(full_file_name, dest)