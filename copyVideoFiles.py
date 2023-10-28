import shutil

# Define the source and destination folders
source_folder = './Dataset_CVDLPT_Videos_Blurred'  # Replace with the path to your source folder
destination_folder = './Dataset_CVDLPT_Videos_Blurred_WErrors'  # Replace with the path to your destination folder

# Define the path to the text file containing filenames
file_list_path = 'copyfiles.txt'  # Replace with the path to your text file

# Open the text file and read the filenames
with open(file_list_path, 'r') as file:
    filenames = file.read().splitlines()


# Iterate through the list of filenames and copy them
for item in filenames:
    file=item.split('_')
    filename=f"E{file[0][1]}_P{file[1][1]}_T{file[2][1]}_C{file[3][1]}_anonymized.mp4"
    #filename=f"E{file[0][0]}_P{file[0][1]}_T{file[0][2]}_C{file[0][3]}_anonymized.mp4"
    #filename=f"E{file[0]}_P{file[1]}_T{file[2]}_C{file[3]}_anonymized.mp4"
    source_file = f'{source_folder}/{filename}'
    destination_file = f'{destination_folder}/{filename}'
    #print(filename)
    #print(file)
    
    try:
        shutil.copy(source_file, destination_file)
        print(f'Copied: {source_file}')
    except FileNotFoundError:
        print(f'File not found: {source_file}')
    except shutil.SameFileError:
        print(f'Source and destination are the same for: {source_file}')
    except Exception as e:
        print(f'Error copying {source_file}: {e}')
    
