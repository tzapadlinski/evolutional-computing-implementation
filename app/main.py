import tkinter as tk
from gui.gui import GeneticAlgorithmGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()