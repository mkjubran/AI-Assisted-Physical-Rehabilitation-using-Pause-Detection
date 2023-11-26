import os
import subprocess
import time
import pdb
from docx import Document
from datetime import datetime

# Specify the folder path containing long mp4 files
mp4_source_folder_path = '../Dataset_CVDLPT_Videos'

# Specify the folder path to output mp4 segments
mp4_destination_folder_path = '../Dataset_CVDLPT_Videos_Segments_11_2023'

# Specify the folder path containing delination files
delination_source_folder_path = '../Dataset_CVDLPT_Videos_Blurred_SegmentsDelineation_11_2023'

# Initialize an empty list to store numbers
numbers_list = []

# to read text file, verify the content, and return the content as a numpy array
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []

    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)

    return full_text

# Loop through files in the folder
for filename in os.listdir(delination_source_folder_path):
    if filename.endswith('.docx'):  # Check if the file is a text file
        #print(f"{filename.split('_3D')[0]}.mp4")
        delination_source_file_path = os.path.join(delination_source_folder_path,filename)
        mp4_source_file_path = os.path.join(mp4_source_folder_path,f"{filename.split('_anonymized')[0]}.mp4")
        #destination_file_path = os.path.join(destination_folder_path,filename)
        if os.path.exists(delination_source_file_path):
             #print(delination_source_file_path)
             #print(mp4_source_file_path)
             print(read_docx(delination_source_file_path))
             if os.path.exists(mp4_source_file_path):
                   #if f"{filename.split('_anonymized')[0]}.mp4" == "E7_P0_T2_C1.mp4":
                   #print(mp4_source_file_path)
                   content = read_docx(delination_source_file_path)
                   print(content)
                   FnumPrev=0
                   content = [item for item in content if item != ""]
                   #pdb.set_trace()
                   for cnti in range(len(content)):
                       Fnums = [int(x) for x in content[cnti].split()]
                       if len(Fnums) == 2 and isinstance(Fnums[0], int) and isinstance(Fnums[1], int) and FnumPrev <= Fnums[0] and Fnums[1] > Fnums[0]:
                          FnumPrev=Fnums[0]
                          mp4_destination_file_path = os.path.join(mp4_destination_folder_path,f"{filename.split('_anonymized')[0]}_seg{cnti}.mp4")
                          print(Fnums,Fnums[0],Fnums[1])
                          print(mp4_destination_file_path)

                          # Define the FFmpeg command as a list of strings
                          ffmpeg_command = [
                              'ffmpeg',
                              '-y',
                              '-i', mp4_source_file_path,  # Input video file
                              '-c:v', 'libx264',  # Video codec
                              '-vf', f"select=between(n\,{Fnums[0]}\,{Fnums[1]}),setpts=PTS-STARTPTS",
                              '-q:v', "2",  # Output quality
                              '-an',
                              mp4_destination_file_path  # Output video file
                          ]
                          # Execute the FFmpeg command
                          try:
                              #print(ffmpeg_command)
                              subprocess.run(ffmpeg_command, check=True)
                          except subprocess.CalledProcessError as e:
                              current_datetime = datetime.now()
                              formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
                              with open('error_log.txt', 'a') as file:
                                   file.writelines(f"{formatted_datetime}\n")
                                   file.writelines(f"{delination_source_file_path}\n")
                                   file.writelines(f"{mp4_destination_file_path}\n")
                                   file.writelines(f"{ffmpeg_command}\n")
                                   file.writelines(f"--------------------------------\n")
                              print("Error:", e)

                          #print(Fnums[cnti],Fnums[cnti+1])
                          #print(mp4_destination_file_path)

                          #time.sleep(10.5)    # Pause 5.5 seconds
                          
                       else:
                          print(delination_source_file_path)
                          pdb.set_trace() 
