import os
import json
import csv
from typing import List, Any, Dict, Tuple
from datetime import datetime
import numpy as np

SATELLITE_PATH = 'satellites'

class CSV_Manager:
    def __init__(self, directory: str = "trash"):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_csv(self, filename: str, labels: List[str], data: List[List[Any]]) -> str:
        """
        Save a CSV file with varying column lengths. Pads shorter columns with empty strings and ensures 
        labels cover all columns.

        Args:
            filename (str): The name of the CSV file (without '.csv' extension).
            labels (List[str]): Column headers.
            data (List[List[Any]]): Data to save, structured as a list of columns.

        Returns:
            str: The name of the saved CSV file.
        """
        filename += ".csv"
        filepath = os.path.join(self.directory, filename)

        # Determine the maximum number of rows and columns
        rows = max(len(column) for column in data)
        num_columns = len(data)

        # Pad shorter columns with empty strings
        for column in data:
            column.extend([""] * (rows - len(column)))

        # Pad labels if there are not enough headers
        if len(labels) < num_columns:
            labels.extend([""] * (num_columns - len(labels)))

        # Write the CSV file
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the column headers (labels)
            writer.writerow(labels)

            # Write the data row by row
            for index in range(rows):
                writer.writerow([str(data[column][index]) for column in range(num_columns)])

        return filename

    def load_csv(self, filename: str) -> Tuple[List[str], List[List[Any]]]:
        """
        Load a CSV file and return the labels and data (excluding metadata).

        Args:
            filename (str): The name of the CSV file to load.

        Returns:
            Tuple[List[str], List[List[Any]]]: A tuple containing the labels and data where 'data' is a list of columns.
        """
        filepath = os.path.join(self.directory, filename + ".csv")

        with open(filepath, mode='r') as file:
            reader = csv.reader(file)
            labels = next(reader)  # First row is the labels
            data = list(zip(*reader))  # Transpose rows to columns

        return labels, [list(column) for column in data]

    def save_with_metadata(self, filename: str, metadata: List[str], labels: List[str], data: List[List[Any]]):
        labels = ["Metadata"] + labels
        data = metadata + data
        self.save_csv(filename, labels, data)

    def load_with_metadata(self, filename: str) -> Tuple[List[str], List[List[Any]], List[Any]]:
        """
        Load a CSV file with metadata in the first column, and return labels, data, and metadata.

        Args:
            filename (str): The name of the CSV file to load.

        Returns:
            Tuple[List[str], List[List[Any]], List[Any]]: The labels, data, and metadata extracted from the first column.
        """
        labels, data = self.load_csv(filename)

        # Extract metadata (first column)
        metadata = data[0]
        data = data[1:]  # Remove the first column from data
        labels = labels[1:]  # Remove the first label corresponding to the metadata

        return labels, data, metadata

class JSON_Manager:
    def __init__(self, directory: str = "trash"):
        """
        Initialize the JSON_Manager with a directory to save/load JSON files.

        Args:
            directory (str): The directory where JSON files are saved/loaded.
        """
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def file_exists(self, filename: str) -> bool:
        """
        Check if a JSON file exists in the directory.

        Args:
            filename (str): The name of the file (without extension).

        Returns:
            bool: True if the file exists, False otherwise.
        """
        filename += ".json"
        filepath = os.path.join(self.directory, filename)
        return os.path.exists(filepath)

    def save_json(self, filename: str, data: Any) -> str:
        """
        Save data to a JSON file.

        Args:
            filename (str): The name of the JSON file (without '.json' extension).
            data (Any): The data to be saved.

        Returns:
            str: The path of the saved JSON file.
        """
        filename += '.json'
        filepath = os.path.join(self.directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved data to {filepath}")
        return filepath

    def save_with_metadata(self, filename: str, metadata: Dict[str, Any], data: Any) -> str:
        """
        Save data with metadata to a JSON file.

        Args:
            filename (str): The name of the JSON file (without '.json' extension).
            metadata (Dict[str, Any]): Metadata to be saved.
            data (Any): Data to be saved.

        Returns:
            str: The path of the saved JSON file.
        """
        json_object = {
            "metadata": metadata,
            "data": data
        }

        return self.save_json(filename, json_object)

    def load_json(self, filename: str, quiet: bool = False) -> Any:
        """
        Load data from a JSON file.

        Args:
            filename (str): The name of the JSON file (without '.json' extension).

        Returns:
            Any: The loaded data.
        """
        filepath = os.path.join(self.directory, filename + ".json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not quiet:
            print(f"Loaded data from {filepath}")
        return data

    def load_with_metadata(self, filename: str, quiet: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Load data and metadata from a JSON file by calling load_json, and then separate them.

        Args:
            filename (str): The name of the JSON file (without '.json' extension).

        Returns:
            Tuple[Any, Dict[str, Any]]: A tuple containing the data and metadata.
        """
        json_object = self.load_json(filename, quiet=quiet)

        # Separate the data and metadata
        data = json_object.get("data", None)
        metadata = json_object.get("metadata", None)

        return data, metadata

class SatelliteFileManager:
    def __init__(self, json_manager: JSON_Manager = None):
        """
        Initialize the SatelliteFileManager with an instance of JSON_Manager to handle file operations.

        Args:
            json_manager (JSON_Manager): An instance of JSON_Manager for saving/loading satellites.
        """
        if json_manager is None:
            self.json_manager = JSON_Manager("satellites")  # Use a default path
        else:
            self.json_manager = json_manager

    def satellite_file_exists(self, filename: str) -> bool:
        """
        Check if a satellite file exists by using the JSON_Manager's file_exists method.

        Args:
            filename (str): The name of the file (without extension).

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.json_manager.file_exists(filename)

    def load_satellites(self, filename: str, quiet: bool = False) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load satellites from a JSON file along with metadata and convert satellite data to floats.

        Args:
            filename (str): The name of the file (without '.json' extension).
            quiet (bool): If True, suppress output.

        Returns:
            Tuple[List[Dict], Dict[str, Any]]: A list of satellite configurations and metadata.
        """
        data, metadata = self.json_manager.load_with_metadata(filename)
        
        # Convert all satellite data to floats
        if isinstance(data, dict) and "satellite_pool" in data:
            num_sat = len(data["satellite_pool"])
            data["satellite_pool"] = [[float(value) for value in satellite] for satellite in data["satellite_pool"]]
        else:
            num_sat = len(data)
            data = [[float(value) for value in satellite] for satellite in data]

        if not quiet:
            print(f"Loaded {num_sat} satellites from {filename}.json with metadata: {metadata}")
        return data, metadata

    def save_satellites(self, filename: str, satellites: List[Dict], metadata: Dict[str, Any] = None, quiet: bool = False) -> None:
        """
        Save satellite configurations to a JSON file with metadata, including the current timestamp.

        Args:
            satellites (List[Dict]): Satellite configurations to save.
            filename (str): The name of the file (without '.json' extension).
            metadata (Dict[str, Any]): Additional metadata to save.
            quiet (bool): If True, suppress output.

        Returns:
            None
        """
        # Add a timestamp to the metadata
        if metadata is None:
            metadata = {}
        metadata['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save satellites with metadata
        self.json_manager.save_with_metadata(filename, metadata, satellites)

class ChromosomeFileManager:
    def __init__(self, directory: str = "chromosomes", format: str = 'json'):
        """
        Initialize the ChromosomeFileManager.

        Args:
            directory (str): Directory to save/load chromosome files.
            format (str): File format to use ('json' or 'csv').
        """
        self.format = format.lower()
        if self.format == 'json':
            self.manager = JSON_Manager(directory)
        elif self.format == 'csv':
            self.manager = CSV_Manager(directory)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'.")

    def save_chromosome(self, chromosome: List[float], metadata: Dict[str, Any], filename: str) -> None:
        """
        Save a single chromosome with metadata.

        Args:
            chromosome (List[float]): Chromosome data.
            metadata (Dict[str, Any]): Metadata associated with the chromosome.
            filename (str): Name of the file (without extension).
        """
        if self.format == 'json':
            data = chromosome
            self.manager.save_with_metadata(filename, metadata, data)
        elif self.format == 'csv':
            # For CSV, store chromosome as a single column
            labels = ['Chromosome']
            data = [chromosome]
            # Since CSV_Manager expects columns, each chromosome is a column
            # However, chromosomes have varying lengths, so pad them
            self.manager.save_with_metadata(filename, [json.dumps(metadata)], labels, data)

    def load_chromosome(self, filename: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Load a single chromosome with metadata.

        Args:
            filename (str): Name of the file (without extension).

        Returns:
            Tuple[List[float], Dict[str, Any]]: Chromosome data and its metadata.
        """
        if self.format == 'json':
            data, metadata = self.manager.load_with_metadata(filename)
            return data, metadata
        elif self.format == 'csv':
            labels, data, metadata_json = self.manager.load_with_metadata(filename)
            # Assuming metadata was saved as JSON string in the 'Metadata' column
            metadata = json.loads(metadata_json[0]) if metadata_json else {}
            chromosome = [float(x) for x in data[0] if x != ""]
            return chromosome, metadata

class PopulationFileManager:
    def __init__(self, directory: str = "populations", format: str = 'json', population_size: int = 100):
        """
        Initialize the PopulationFileManager.

        Args:
            directory (str): Directory to save/load population files.
            format (str): File format to use ('json' or 'csv').
            population_size (int): Number of chromosomes per population.
        """
        self.format = format.lower()
        self.population_size = population_size
        if self.format == 'json':
            self.manager = JSON_Manager(directory)
        elif self.format == 'csv':
            self.manager = CSV_Manager(directory)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'.")

    def file_exists(self, filename: str) -> bool:
        """
        Check if a population file already exists.

        Args:
            filename (str): Name of the file (without extension).

        Returns:
            bool: True if the file exists, False otherwise.
        """
        if self.format == 'json':
            filepath = f"{self.manager.directory}/{filename}.json"
        elif self.format == 'csv':
            filepath = f"{self.manager.directory}/{filename}.csv"
        return os.path.exists(filepath)

    def save_population(self, population: List[List[float]], metadata: Dict[str, Any], filename: str) -> None:
        """
        Save a population of chromosomes with metadata.

        Args:
            population (List[List[float]]): List of chromosomes.
            metadata (Dict[str, Any]): Metadata associated with the population.
            filename (str): Name of the population file (without extension).
        """
        if self.format == 'json':
            data = population
            self.manager.save_with_metadata(filename, metadata, data)
        elif self.format == 'csv':
            # For CSV, each chromosome is a column
            labels = [f'Chromosome_{i+1}' for i in range(len(population))]
            # Transpose the population to have chromosomes as columns
            # and pad shorter chromosomes
            max_length = max(len(chrom) for chrom in population)
            data = [chrom + [""] * (max_length - len(chrom)) for chrom in population]
            # Transpose data to have rows
            data_transposed = list(map(list, zip(*data)))
            # Save with metadata
            # Since CSV_Manager expects metadata as a list of strings,
            # we serialize metadata as a JSON string
            metadata_serialized = [json.dumps(metadata)]
            self.manager.save_with_metadata(filename, metadata_serialized, labels, data_transposed)

    def load_population(self, filename: str) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Load a population of chromosomes with metadata.

        Args:
            filename (str): Name of the population file (without extension).

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]: Population data and its metadata.
        """
        if self.format == 'json':
            data, metadata = self.manager.load_with_metadata(filename)
            return data, metadata
        elif self.format == 'csv':
            labels, data_rows, metadata_json = self.manager.load_with_metadata(filename)
            # Deserialize metadata
            metadata = json.loads(metadata_json[0]) if metadata_json else {}
            # Transpose rows back to chromosomes
            chromosomes = []
            for column in data_rows:
                chrom = [float(x) for x in column if x != ""]
                chromosomes.append(chrom)
            return chromosomes, metadata

    def save_multiple_populations(self, populations: List[List[List[float]]], metadatas: List[Dict[str, Any]], base_filename: str = "population") -> None:
        """
        Save multiple populations.

        Args:
            populations (List[List[List[float]]]): List of populations, each being a list of chromosomes.
            metadatas (List[Dict[str, Any]]): List of metadata dictionaries for each population.
            base_filename (str): Base name for population files. Files will be named as base_filename_1, base_filename_2, etc.
        """
        for idx, (population, metadata) in enumerate(zip(populations, metadatas), start=1):
            filename = f"{base_filename}_{idx}"
            self.save_population(population, metadata, filename)

    def load_multiple_populations(self, base_filename: str = "population", count: int = 10) -> List[Tuple[List[List[float]], Dict[str, Any]]]:
        """
        Load multiple populations.

        Args:
            base_filename (str): Base name for population files. Files should be named as base_filename_1, base_filename_2, etc.
            count (int): Number of populations to load.

        Returns:
            List[Tuple[List[List[float]], Dict[str, Any]]]: List of populations and their metadata.
        """
        populations = []
        for idx in range(1, count + 1):
            filename = f"{base_filename}_{idx}"
            try:
                population, metadata = self.load_population(filename)
                populations.append((population, metadata))
            except FileNotFoundError:
                print(f"Population file {filename} not found.")
        return populations


import orjson
import hashlib

# Ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Convert numpy types to native Python types for JSON serialization
def convert_numpy_to_native(data):
    if isinstance(data, dict):
        return {k: convert_numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        return data.item()  # Convert numpy types to native Python types
    else:
        return data

# Save results with metadata using orjson and handle numpy types
def save_results_with_metadata(filename, metadata, results):
    # Convert numpy types in the results to native Python types
    results = convert_numpy_to_native(results)
    
    # Add current datetime before generating the metadata_id
    metadata["time_saved"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current save time

    # Generate unique metadata_id after datetime has been added
    metadata_id = generate_metadata_id(metadata)
    metadata["metadata_id"] = metadata_id  # Add unique metadata_id

    # Save results with updated metadata
    with open(f"{filename}.json", 'wb') as f:  # open in binary mode for orjson
        data = {"metadata": metadata, "results": results}
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

# Save results without metadata, calls save_results_with_metadata
def save_results(filename, results):
    # Just calling save_results_with_metadata with empty metadata
    save_results_with_metadata(filename, metadata={}, results=results)

# Load results and metadata using orjson
def load_results_with_metadata(filename):
    with open(f"{filename}.json", 'rb') as f:  # open in binary mode for orjson
        data = orjson.loads(f.read())
    
    if "metadata" in data:
        return data["results"], data["metadata"]
    else:
        return data, {}

# Load only results, calls load_results_with_metadata
def load_results(filename):
    results, _ = load_results_with_metadata(filename)
    return results

# Check if a results file exists
def file_exists(filename):
    return os.path.exists(f"{filename}.json")

# Function to save plot metadata using orjson
def save_plot_metadata(filename, metadata):
    # Add the current time internally before saving
    metadata["time_saved"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save the metadata to a JSON file
    with open(f"{filename}.json", 'wb') as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

# Function to load plot metadata
def load_plot_metadata(filename):
    if os.path.exists(f"{filename}.json"):
        with open(f"{filename}.json", 'rb') as f:
            return orjson.loads(f.read())
    return None

# Generate a unique metadata ID based on a hash of the metadata content (or datetime)
def generate_metadata_id(metadata):
    # Create a hash from the current datetime for uniqueness
    metadata_string = orjson.dumps(metadata).decode('utf-8')
    return hashlib.md5(metadata_string.encode('utf-8')).hexdigest()

# Function to check if plot was generated from current parent data
def is_generated_from(child_metadata, parent_metadata):
    if child_metadata == None or parent_metadata == None:
        return False
    
    return child_metadata.get("generated_from") == parent_metadata.get("metadata_id")
