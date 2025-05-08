import random
import numpy as np

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def check_min_distance(matrix_list, min_distance=20):
    num_points = len(matrix_list)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if distance(matrix_list[i], matrix_list[j]) < min_distance:
                return False  # Too close
    return True  # All distances OK

success = False
while not success:
    # Generate 10 random base points
    random_list_1 = [random.uniform(-200, 200) for _ in range(10)]
    random_list_2 = [random.uniform(-200, 200) for _ in range(10)]
    random_list_3 = [random.uniform(0, 0) for _ in range(10)]  # Fixed z-plane

    matrix_list = [[x, y, z] for x, y, z in zip(random_list_1, random_list_2, random_list_3)]
    success = check_min_distance(matrix_list)

# Add a second bead 40 units higher in z
matrix_list_2 = [[x, y, z + 35] for x, y, z in matrix_list]

# Interleave both sets
interleaved_matrix = []
for a, b in zip(matrix_list, matrix_list_2):
    interleaved_matrix.append(a)
    interleaved_matrix.append(b)

# Save to file
with open("random_position_output.txt", "w") as file:
    for row in interleaved_matrix:
        file.write(" ".join(map(str, row)) + "\n")

print("Saved interleaved matrix with all dimers at least 20 units apart.")
