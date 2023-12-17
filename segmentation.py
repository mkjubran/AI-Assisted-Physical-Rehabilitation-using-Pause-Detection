import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Load the .npz file
data3 = np.load('E0_P0_T0_C0_3D.npz')
data2 = np.load('E0_P0_T0_C0_2D.npz')

array2 = data2['reconstruction']
array3 = data3['reconstruction']

# Sample energy calculation function
def calculate_energy(frame):
    # Perform energy calculation based on the frame
    energy = np.sum((array3[0][0] - array3[0][frame])**2)  # Replace with your own calculation
    #print("Energy at frame", frame, "=", energy)
    return energy

# Number of frames and frames per exercise
num_frames = len(array3[0])
frames_per_exercise = 70

# Lists to store frame and energy values
frames = []
energies = []

# Count the number of exercises
num_exercises = num_frames // frames_per_exercise

# Generate frames and calculate energy
for exercise in range(num_exercises):
    arrays = []
    for frame in range(frames_per_exercise):
        frame_num = exercise * frames_per_exercise + frame
        frames.append(frame_num)
        energy = calculate_energy(frame_num)
        energies.append(energy)
        arrays.append(array3[0][frame])

    # Plotting the energy values
    plt.plot(frames, energies, marker='o')

    # Adding labels and title
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.title('Segmentation - Exercise {}'.format(exercise + 1))

    # Displaying the plot
    plt.show()

    # Clear the lists for the next exercise
    frames.clear()
    energies.clear()


    # Print the exercise number
    print("----------------------------------------Exercise----------------------------------------", exercise + 1)

# Print the total number of exercises
print("Total number of exercises:", num_exercises)