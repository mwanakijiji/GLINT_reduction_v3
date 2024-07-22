import tkinter as tk
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from modules import functions
from tkinter import filedialog
import glob

def run_command_1():
    command = ["echo", "command_1"]
    subprocess.run(command)

def run_command_2():
    command = ["echo", "command_2"]
    subprocess.run(command)

def run_command_3():
    functions.the_fcn()

# Drop-down menu
def find_directory():
    directory = filedialog.askdirectory()
    selected_option.set(directory)

def refresh_empty_space_1():
    directory = selected_option.get()
    files = glob.glob(directory + "/*")
    file_list = "\n".join(files)
    empty_space_1.delete(1.0, tk.END)  # Clear the current content
    empty_space_1.insert(tk.END, file_list)  # Insert the new content

parser = argparse.ArgumentParser()
parser.add_argument("--no-window", action="store_true", help="Do not display the window")
args = parser.parse_args()

if not args.no_window:
    window = tk.Tk()
    window.resizable(True, True)

    # Top row
    top_row = tk.Frame(window)
    top_row.pack()


    selected_option = tk.StringVar()
    selected_option.set("")
    display_menu = tk.Label(top_row, textvariable=selected_option)
    display_menu.pack(side=tk.TOP)

    find_button = tk.Button(top_row, text="Image Parent Directory", command=find_directory)
    find_button.pack(side=tk.TOP)

    file_names_label = tk.Label(top_row, text="File names")
    file_names_label.pack(side=tk.TOP)
    
    empty_space_1 = tk.Text(top_row, width=20, height=10)
    selected_option.trace("w", lambda *args: refresh_empty_space_1())  # Call refresh_empty_space_1 when selected_option changes
    empty_space_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar_1 = tk.Scrollbar(top_row, command=empty_space_1.yview)
    empty_space_1.configure(yscrollcommand=scrollbar_1.set)
    empty_space_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_1.pack(side=tk.RIGHT, fill=tk.Y)

    empty_space_2 = tk.Text(top_row, width=20, height=10)
    empty_space_2.insert(tk.END, "Placeholder for stuff being printed to log")
    empty_space_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar_2 = tk.Scrollbar(top_row, command=empty_space_2.yview)
    empty_space_2.configure(yscrollcommand=scrollbar_2.set)
    scrollbar_2.pack(side=tk.RIGHT, fill=tk.Y)

    empty_space_3 = tk.Canvas(top_row, width=256, height=256)
    empty_space_3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Generate a 256x256 array of random numbers
    random_array = np.random.rand(256, 256)

    # Display the array as an image
    plt.clf()
    plt.imshow(random_array, cmap='gray')
    plt.axis('off')  # Hide the axis
    plt.savefig('random_image.png')  # Save the image as a file

    # Load the image file and display it on the canvas
    image = tk.PhotoImage(file='random_image.png')
    empty_space_3.create_image(128, 128, anchor=tk.CENTER, image=image)
    empty_space_3.configure(width=256, height=256)

    scrollbar_3 = tk.Scrollbar(top_row, command=empty_space_3.yview)
    empty_space_3.configure(yscrollcommand=scrollbar_3.set)
    scrollbar_3.pack(side=tk.RIGHT, fill=tk.Y)

    # Bottom row
    bottom_row = tk.Frame(window)
    bottom_row.pack()

    button_1 = tk.Button(bottom_row, text="Primitives", command=run_command_1)
    button_1.pack(side=tk.LEFT)
    # Drop-down menu
    menu = tk.Menu(window, tearoff=0)
    menu.add_command(label="Option 1")
    menu.add_command(label="Option 2")
    menu.add_command(label="Option 3")
    def open_menu(event):
        menu.post(event.x_root, event.y_root)
    button_1.bind("<Button-1>", open_menu)

    button_2 = tk.Button(bottom_row, text="Command 2", command=run_command_2)
    button_2.pack(side=tk.LEFT)

    button_3 = tk.Button(bottom_row, text="Command 3", command=run_command_3)
    button_3.pack(side=tk.LEFT)

    window.mainloop()