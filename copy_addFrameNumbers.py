import os
import subprocess
import time

# Specify the folder path containing mp4 files
source_folder_path = '../Dataset_CVDLPT_Videos_Blurred'

# Specify the folder path to output mp4 files with frame numbers
destination_folder_path = '../Dataset_CVDLPT_Videos_Blurred_WFrameNum'

# Initialize an empty list to store numbers
numbers_list = []

# Loop through files in the folder
for filename in os.listdir(source_folder_path):
    if filename.endswith('.mp4'):  # Check if the file is a text file
        source_file_path = os.path.join(source_folder_path,filename)
        destination_file_path = os.path.join(destination_folder_path,filename)
        if os.path.exists(source_file_path):
              # Define the FFmpeg command as a list of strings
              ffmpeg_command = [
                   'ffmpeg',
                   '-y',
                   '-i', source_file_path,  # Input video file
                   '-c:v', 'libx264',  # Video codec
                   '-vf', "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=40: box=1: boxcolor=white: boxborderw=5",
                   '-q:v', "2",  # Output quality
                   destination_file_path  # Output video file
              ]
              # Execute the FFmpeg command
              try:
                   #print(ffmpeg_command)
                   subprocess.run(ffmpeg_command, check=True)
              except subprocess.CalledProcessError as e:
                   print("Error:", e)
              #time.sleep(5.5)    # Pause 5.5 seconds
