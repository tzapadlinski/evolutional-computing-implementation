import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
from app.evolutionary.algorithm import EvolutionaryAlgorithm
from app.evolutionary.function import Function

class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Parameters")
        self.root.geometry("700x1000")
        self.root.configure(bg="#f0f0f5")

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="Genetic Algorithm Configuration",
                               font=("Helvetica", 16, "bold"), bg="#f0f0f5")
        title_label.pack(pady=10)

        input_frame = ttk.Frame(self.root, padding="10 10 10 10")
        input_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(input_frame, text="Population Size:", font=("Helvetica", 11)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.population_size_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.population_size_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of Variables:", font=("Helvetica", 11)).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_variables_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.num_variables_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of Epochs:", font=("Helvetica", 11)).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.num_epochs_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.num_epochs_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Beginning of Range:", font=("Helvetica", 11)).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.begin_range_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.begin_range_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="End of Range:", font=("Helvetica", 11)).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.end_range_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.end_range_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Selection Method:", font=("Helvetica", 11)).grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.selection_method_var = tk.StringVar(value="roulette")
        self.selection_method_dropdown = ttk.Combobox(input_frame, textvariable=self.selection_method_var, font=("Helvetica", 11), state="readonly")
        self.selection_method_dropdown['values'] = ("roulette", "tournament", "best")
        self.selection_method_dropdown.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Elite Strategy Amount:", font=("Helvetica", 11)).grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.elite_strategy_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.elite_strategy_entry.grid(row=6, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Cross Probability:", font=("Helvetica", 11)).grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.cross_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.cross_prob_entry.grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Mutation Probability:", font=("Helvetica", 11)).grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.mutation_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.mutation_prob_entry.grid(row=8, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Inversion Probability:", font=("Helvetica", 11)).grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.inversion_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.inversion_prob_entry.grid(row=9, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Uniform Crossover Probability:", font=("Helvetica", 11)).grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.uniform_crossover_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.uniform_crossover_prob_entry.grid(row=9, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Cross Method:", font=("Helvetica", 11)).grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.cross_method_var = tk.StringVar(value="one-point")
        self.cross_method_dropdown = ttk.Combobox(input_frame, textvariable=self.cross_method_var, font=("Helvetica", 11), state="readonly")
        self.cross_method_dropdown['values'] = ("one-point", "two-point", "three-point", "homo")
        self.cross_method_dropdown.grid(row=10, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Mutation Method:", font=("Helvetica", 11)).grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.mutation_method_var = tk.StringVar(value="one-point")
        self.mutation_method_dropdown = ttk.Combobox(input_frame, textvariable=self.mutation_method_var, font=("Helvetica", 11), state="readonly")
        self.mutation_method_dropdown['values'] = ("one-point", "two-point")
        self.mutation_method_dropdown.grid(row=11, column=1, padx=5, pady=5)

        available_functions = Function.get_available_functions()
        ttk.Label(input_frame, text="Function to optimize:", font=("Helvetica", 11)).grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.function_var = tk.StringVar(value=available_functions[0])
        self.function_dropdown = ttk.Combobox(input_frame, textvariable=self.function_var, font=("Helvetica", 11), state="readonly")
        self.function_dropdown['values'] = available_functions
        self.function_dropdown.grid(row=11, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Optimization mode:", font=("Helvetica", 11)).grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.optimization_mode_var = tk.StringVar(value="MAX")
        self.optimization_mode_dropdown = ttk.Combobox(input_frame, textvariable=self.mutation_method_var, font=("Helvetica", 11), state="readonly")
        self.optimization_mode_dropdown['values'] = ("MAX", "MIN")
        self.optimization_mode_dropdown.grid(row=11, column=1, padx=5, pady=5)

        run_button = ttk.Button(self.root, text="Run Algorithm", command=self.run_algorithm, style="Accent.TButton")
        run_button.pack(pady=15)

        output_frame = ttk.LabelFrame(self.root, text="Output", padding="10 10 10 10")
        output_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.output_text = tk.Text(output_frame, wrap="word", font=("Helvetica", 10), height=10, width=40, bg="#ffffff", fg="#333333")
        self.output_text.pack(expand=True, fill="both", padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.output_text, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        style = ttk.Style(self.root)
        style.configure("Accent.TButton", font=("Helvetica", 11), background="#5a9", foreground="#fff")

    def run_algorithm(self):
        try:
            population_size = int(self.population_size_entry.get())
            num_variables = int(self.num_variables_entry.get())
            num_epochs = int(self.num_epochs_entry.get())
            begin_range = float(self.begin_range_entry.get())
            end_range = float(self.end_range_entry.get())
            elite_strategy_amount = float(self.elite_strategy_entry.get())
            cross_prob = float(self.cross_prob_entry.get())
            mutation_prob = float(self.mutation_prob_entry.get())
            inversion_prob = float(self.inversion_prob_entry.get())
            uniform_crossover = float(self.uniform_crossover_prob_entry.get())
            selection_method = self.selection_method_var.get()
            cross_method = self.cross_method_var.get()
            mutation_method = self.mutation_method_var.get()
            function = self.function_var.get()
            optimization_mode = self.optimization_mode_var.get()

            start_time = time.time()

            alg = EvolutionaryAlgorithm(
                population_size = population_size, 
                chromosome_size = num_variables,
                generations = num_epochs,
                p_mutation = mutation_prob,
                p_crossover = cross_prob,
                selection_method = selection_method,
                mutation_method = mutation_method,
                crossover_method = cross_method,
                function = function,
                optimization_mode = optimiziation_mode,
                p_uniform = uniform_crossover,
                lower_bound = begin_range,
                upper_bound = end_range,
                p_inversion = inversion_prob)
            
            alg.run()

            result_to_display = 'replace with value to display'

            end_time = time.time()
            execution_time = end_time - start_time

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Running with:\n"
                                           f"Population Size: {population_size}\n"
                                           f"Number of Variables: {num_variables}\n"
                                           f"Number of Epochs: {num_epochs}\n"
                                           f"Range: {begin_range} to {end_range}\n"
                                           f"Selection Method: {selection_method}\n"
                                           f"Elite Strategy Amount: {elite_strategy_amount}\n"
                                           f"Cross Probability: {cross_prob}\n"
                                           f"Mutation Probability: {mutation_prob}\n"
                                           f"Inversion Probability: {inversion_prob}\n"
                                           f"Uniform Crossover Probability: {uniform_crossover}\n"
                                           f"Cross Method: {cross_method}\n"
                                           f"Mutation Method: {mutation_method}\n"
                                           f"Function to optimize: {function}\n"
                                           f"Optimization mode: {optimization_mode}\n")

            self.output_text.insert(tk.END, f"\nExecution Time: {execution_time:.4f} seconds\n")
            self.output_text.insert(tk.END, "\nResult: [result_to_display]")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid values for all input fields.")