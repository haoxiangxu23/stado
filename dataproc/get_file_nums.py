import os

def get_number(path):
    files = [f for f in os.listdir(path) if f.endswith(".png")]
    return len(files)

# print(get_number("MOC-Detector/data/JHMDB/Frames/jump/Stadium_Plyometric_Workout_jump_f_cm_np1_ba_bad_6/"))