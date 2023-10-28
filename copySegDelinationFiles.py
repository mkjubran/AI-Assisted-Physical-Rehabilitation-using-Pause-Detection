import shutil
import os
import subprocess

# Define the source and destination folders
source_folder = './Dataset_CVDLPT_Videos_Blurred_SegmentsDelineation'  # Replace with the path to your source folder
destination_folder = './Dataset_CVDLPT_Videos_Blurred_SegmentsDelineation_WErrors'  # Replace with the path to your destination folder

list_folder = './Dataset_CVDLPT_Videos_Blurred_WErrors'

# Loop through files in the folder
for filename in os.listdir(list_folder):
    if filename.endswith('.mp4'):  # Check if the file is a text file
        filenametxt=f"{filename.split('.')[0]}_3D_seg.txt"
        source_file=f"{source_folder}/{filenametxt}"
        destination_file=f"{destination_folder}/{filenametxt}"
        try:
           shutil.copy(source_file, destination_file)
           print(f'Copied: {source_file}')
        except FileNotFoundError:
           print(f'File not found: {source_file}')
        except shutil.SameFileError:
           print(f'Source and destination are the same for: {source_file}')
        except Exception as e:
           print(f'Error copying {source_file}: {e}')


#E2_P3_T1_C0_anonymized_3D_seg.txt
#E2_P6_T1_C1_anonymized.mp4
