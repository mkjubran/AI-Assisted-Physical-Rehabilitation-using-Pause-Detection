import os
import subprocess
import time
import pdb

# Specify the folder path containing long mp4 files
mp4_source_folder_path = '../Dataset_CVDLPT_Videos'

# Specify the folder path to output mp4 segments
mp4_destination_folder_path = '../Dataset_CVDLPT_Videos_Segments'

# Specify the folder path containing delination files
delination_source_folder_path = '../Dataset_CVDLPT_Videos_Blurred_SegmentsDelineation'

# Initialize an empty list to store numbers
numbers_list = []

# Loop through files in the folder
for filename in os.listdir(delination_source_folder_path):
    if filename.endswith('.txt'):  # Check if the file is a text file
        #print(f"{filename.split('_3D')[0]}.mp4")
        delination_source_file_path = os.path.join(delination_source_folder_path,filename)
        mp4_source_file_path = os.path.join(mp4_source_folder_path,f"{filename.split('_anonymized')[0]}.mp4")
        #destination_file_path = os.path.join(destination_folder_path,filename)
        if os.path.exists(delination_source_file_path):
             #print(delination_source_file_path)
             #print(mp4_source_file_path)
             if os.path.exists(mp4_source_file_path):
                   #print(mp4_source_file_path)
                   with open(delination_source_file_path) as f:
                       Fnums=f.readlines()[0].split()
                       #print(Fnums)
                   for cnti in range(len(Fnums)-1):
                       mp4_destination_file_path = os.path.join(mp4_destination_folder_path,f"{filename.split('_anonymized')[0]}_seg{cnti}.mp4")
                       #print(Fnums[cnti],Fnums[cnti+1])
                       #print(mp4_destination_file_path)
                   
                       # Define the FFmpeg command as a list of strings
                       ffmpeg_command = [
                           'ffmpeg',
                           '-y',
                           '-i', mp4_source_file_path,  # Input video file
                           '-c:v', 'libx264',  # Video codec
                           '-vf', f"select=between(n\,{Fnums[cnti]}\,{Fnums[cnti+1]}),setpts=PTS-STARTPTS",
                           '-q:v', "2",  # Output quality
                           '-an',
                           mp4_destination_file_path  # Output video file
                       ]
                       # Execute the FFmpeg command
                       try:
                           #print(ffmpeg_command)
                           subprocess.run(ffmpeg_command, check=True)
                       except subprocess.CalledProcessError as e:
                           print("Error:", e)

                       print(Fnums[cnti],Fnums[cnti+1])
                       print(mp4_destination_file_path)

                       time.sleep(10.5)    # Pause 5.5 seconds
