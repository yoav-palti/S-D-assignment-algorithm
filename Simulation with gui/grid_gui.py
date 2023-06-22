import tkinter as tk

# Create the application window
import matplotlib.pyplot as plt
import numpy as np

in_calculation = False

window = tk.Tk()
window.title("Grid Click")

# Define the grid size
grid_size_x = 50
grid_size_y = 50

# Define the lists to store points
emitter_list = []
sensor_list = []

# Count the points
point_count = 0

# Create canvas for grid
canvas = tk.Canvas(window, width=500, height=500, bg='white')
canvas.pack()

# Draw grid
for i in range(grid_size_x):
    for j in range(grid_size_y):
        canvas.create_rectangle(i * (500 / grid_size_x), j * (500 / grid_size_y),
                                (i + 1) * (500 / grid_size_x), (j + 1) * (500 / grid_size_y))


# Function to handle clicks
def click(event):
    global point_count
    global in_calculation
    a = np.random.randint(-1000, 1000)
    b = np.random.randint(-1000, 1000)
    x = event.x // (500 / grid_size_x)
    y = event.y // (500 / grid_size_y)
    # limit the points that can be added to emitter_list to 5
    if point_count < 5:
        in_calculation = False
        if (2000 * x + a, 2000 * y + b) not in emitter_list:  # avoid duplications
            emitter_list.append((2000 * x + a, 2000 * y + b))
            point_count += 1
            # We'll mark our points with a red circle
            canvas.create_oval(x * (500 / grid_size_x) + (500 / grid_size_x) / 4,
                               y * (500 / grid_size_y) + (500 / grid_size_y) / 4,
                               (x + 1) * (500 / grid_size_x) - (500 / grid_size_x) / 4,
                               (y + 1) * (500 / grid_size_y) - (500 / grid_size_y) / 4, fill="red")
        print("Emitter List: ", emitter_list)
    # allow for additional 8 points to be added to sensor_list
    elif point_count >= 5 and point_count < 13:
        if (2000 * x + a, 2000 * y + b) not in sensor_list:  # avoid duplications
            sensor_list.append((2000 * x + a, 2000 * y + b))
            point_count += 1
            # We'll mark our points with a blue circle
            canvas.create_oval(x * (500 / grid_size_x) + (500 / grid_size_x) / 4,
                               y * (500 / grid_size_y) + (500 / grid_size_y) / 4,
                               (x + 1) * (500 / grid_size_x) - (500 / grid_size_x) / 4,
                               (y + 1) * (500 / grid_size_y) - (500 / grid_size_y) / 4, fill="blue")
        print("Sensor List: ", sensor_list)
    elif not in_calculation:
        grid = np.zeros((2001, 2001))
        in_calculation = True
        pointss = [np.array(emitter_locations) for emitter_locations in emitter_list]
        sensor_locss = [np.array(sensor_locations) for sensor_locations in sensor_list]
        sensor_xs = [sensor_locations[0] for sensor_locations in sensor_list]
        sensor_ys = np.array([sensor_locations[1] for sensor_locations in sensor_list])
        emitter_xs = [emitter_locations[0] for emitter_locations in pointss]
        emitter_ys = np.array([emitter_locations[1] for emitter_locations in pointss])
        print(pointss)
        print(sensor_locss)
        from Testing import Testing
        test_gui = Testing(lambda x: 0)
        sensor_locs, sensor_ts = test_gui.create_assignment_scenario(8, 5, pointss=pointss, sensor_locss=sensor_locss)
        from Assigning import assign_sensor_data
        from Tree_of_Data import TreeOfData
        # assignments, tree = assign_sd(sensor_locs, sensor_ts,400)
        print("Started assigning")
        tree_of_data = TreeOfData(sensor_locs, sensor_ts)
        assignments, emitter_locations, times = tree_of_data.greedy_assigning(10)
        fig, ax = plt.subplots(2, 3)
        fig.delaxes(ax[1, 2])
        for i, assignment in enumerate(assignments):
            assignment_grid, points = tree_of_data.get_heatmap_for_snake(assignment)
            # grid += assignment_grid
            # x = points[0]/2000
            # y = points[1]/2000

            # canvas.create_oval(x * (500 / grid_size_x) + (500 / grid_size_x) / 4,
            #                    y * (500 / grid_size_y) + (500 / grid_size_y) / 4,
            #                    (x + 1) * (500 / grid_size_x) - (500 / grid_size_x) / 4,
            #                    (y + 1) * (500 / grid_size_y) - (500 / grid_size_y) / 4, fill="purple")
            if points[0] != None:
                print(assignment)
                # x = ranged - 2 * ranged / (N - 1) * column
                # y = ranged - 2 * ranged / (N - 1) * row
                ex = np.linspace(0, 100000, 501)
                why = np.linspace(0, 100000, 501)
                ayy = ax[i // 3,i % 3].contourf(ex, why, assignment_grid, 20, cmap='RdGy')
                ax[i // 3,i % 3].scatter(sensor_xs, sensor_ys, c='blue', s=50, alpha=0.5)
                ax[i // 3,i % 3].scatter(emitter_xs, emitter_ys, c='red', s=50, alpha=0.5)
                fig.colorbar(ayy, ax=ax[i // 3,i % 3])
                # plt.imshow(assignment_grid)
        plt.show()


# Bind the click function to the canvas
canvas.bind("<Button-1>", click)

# Start the application
window.mainloop()
