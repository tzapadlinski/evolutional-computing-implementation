import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
from evolutionary.algorithm import EvolutionaryAlgorithm
from evolutionary.function import Function

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

        ttk.Label(input_frame, text="Number of Variables:", font=("Helvetica", 11)).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.num_variables_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.num_variables_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Number of Epochs:", font=("Helvetica", 11)).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.num_epochs_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.num_epochs_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Beginning of Range:", font=("Helvetica", 11)).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.begin_range_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.begin_range_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="End of Range:", font=("Helvetica", 11)).grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.end_range_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.end_range_entry.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Selection Method:", font=("Helvetica", 11)).grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.selection_method_var = tk.StringVar(value="roulette")
        self.selection_method_dropdown = ttk.Combobox(input_frame, textvariable=self.selection_method_var, font=("Helvetica", 11), state="readonly")
        self.selection_method_dropdown['values'] = ("roulette", "tournament", "best")
        self.selection_method_dropdown.grid(row=6, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Elite Strategy Amount:", font=("Helvetica", 11)).grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.elite_strategy_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.elite_strategy_entry.grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Cross Probability:", font=("Helvetica", 11)).grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.cross_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.cross_prob_entry.grid(row=8, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Mutation Probability:", font=("Helvetica", 11)).grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.mutation_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.mutation_prob_entry.grid(row=9, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Inversion Probability:", font=("Helvetica", 11)).grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.inversion_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.inversion_prob_entry.grid(row=10, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Uniform Crossover Probability:", font=("Helvetica", 11)).grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.uniform_crossover_prob_entry = ttk.Entry(input_frame, font=("Helvetica", 11), width=25)
        self.uniform_crossover_prob_entry.grid(row=11, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Cross Method:", font=("Helvetica", 11)).grid(row=12, column=0, padx=5, pady=5, sticky="w")
        self.cross_method_var = tk.StringVar(value="arithmetic")
        self.cross_method_dropdown = ttk.Combobox(input_frame, textvariable=self.cross_method_var, font=("Helvetica", 11), state="readonly")
        self.cross_method_dropdown['values'] = ("arithmetic", "linear", "alpha-blended", "alpha-beta-blended", "averaging")
        self.cross_method_dropdown.grid(row=12, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Mutation Method:", font=("Helvetica", 11)).grid(row=13, column=0, padx=5, pady=5, sticky="w")
        self.mutation_method_var = tk.StringVar(value="uniform")
        self.mutation_method_dropdown = ttk.Combobox(input_frame, textvariable=self.mutation_method_var, font=("Helvetica", 11), state="readonly")
        self.mutation_method_dropdown['values'] = ("uniform", "gaussian")
        self.mutation_method_dropdown.grid(row=13, column=1, padx=5, pady=5)

        available_functions = Function.get_available_functions()
        ttk.Label(input_frame, text="Function to Optimize:", font=("Helvetica", 11)).grid(row=14, column=0, padx=5, pady=5, sticky="w")
        self.function_var = tk.StringVar(value=available_functions[0])
        self.function_dropdown = ttk.Combobox(input_frame, textvariable=self.function_var, font=("Helvetica", 11), state="readonly")
        self.function_dropdown['values'] = available_functions
        self.function_dropdown.grid(row=14, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Optimization Mode:", font=("Helvetica", 11)).grid(row=15, column=0, padx=5, pady=5, sticky="w")
        self.optimization_mode_var = tk.StringVar(value="MAX")
        self.optimization_mode_dropdown = ttk.Combobox(input_frame, textvariable=self.optimization_mode_var, font=("Helvetica", 11), state="readonly")
        self.optimization_mode_dropdown['values'] = ("MAX", "MIN")
        self.optimization_mode_dropdown.grid(row=15, column=1, padx=5, pady=5)

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

    def validate_input(self, field_name, field_value, expected_type,):
        try:
            if expected_type == int:
                value = int(field_value)
            elif expected_type == float:
                value = float(field_value)
            else:
                raise ValueError("Unknown expected type.")

            return value
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid value for {field_name}: {e}")
            return None

    def run_algorithm(self):
        self.output_text.delete(1.0, tk.END)

        population_size = self.validate_input("Population Size", self.population_size_entry.get(), int)
        num_variables = self.validate_input("Number of Variables", self.num_variables_entry.get(), int)
        num_epochs = self.validate_input("Number of Epochs", self.num_epochs_entry.get(), int)
        begin_range = self.validate_input("Begin Range", self.begin_range_entry.get(), float)
        end_range = self.validate_input("End Range", self.end_range_entry.get(), float)
        elite_strategy_amount = self.validate_input("Elite Strategy Amount", self.elite_strategy_entry.get(), float)
        cross_prob = self.validate_input("Cross Probability", self.cross_prob_entry.get(), float)
        mutation_prob = self.validate_input("Mutation Probability", self.mutation_prob_entry.get(), float)
        inversion_prob = self.validate_input("Inversion Probability", self.inversion_prob_entry.get(), float)
        uniform_crossover = self.validate_input("Uniform Crossover Probability", self.uniform_crossover_prob_entry.get(), float)

        selection_method = self.selection_method_var.get()
        cross_method = self.cross_method_var.get()
        mutation_method = self.mutation_method_var.get()
        function = self.function_var.get()
        optimization_mode = self.optimization_mode_var.get()

        if None in [population_size, num_variables, num_epochs, begin_range, end_range, elite_strategy_amount,
                    cross_prob, mutation_prob, inversion_prob, uniform_crossover]:
            return

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
            function = Function(num_variables, function),
            num_variables= num_variables,
            optimization_mode = optimization_mode,
            p_uniform = uniform_crossover,
            lower_bound = begin_range,
            upper_bound = end_range,
            p_inversion = inversion_prob,
            elite_percentage = elite_strategy_amount,
        )

        alg.run()

        execution_time = time.time() - start_time

        self.output_text.insert(tk.END,
                                f"Population Size: {population_size}\n"
                                f"Number of Variables: {num_variables}\n"
                                f"Number of Epochs: {num_epochs}\n"
                                f"Beginning of Range: {begin_range}\n"
                                f"End of Range: {end_range}\n"
                                f"Selection Method: {selection_method}\n"
                                f"Elite Strategy Amount: {elite_strategy_amount}\n"
                                f"Cross Probability: {cross_prob}\n"
                                f"Mutation Probability: {mutation_prob}\n"
                                f"Inversion Probability: {inversion_prob}\n"
                                f"Uniform Crossover Probability: {uniform_crossover}\n"
                                f"Cross Method: {cross_method}\n"
                                f"Mutation Method: {mutation_method}\n"
                                f"Function to Optimize: {function}\n"
                                f"Optimization Mode: {optimization_mode}\n"
                                f"Execution Time: {execution_time:.4f} seconds\n"
                                )