from collections import defaultdict
import re
import torch
from torch_geometric.data import HeteroData
import nltk
from nltk.corpus import stopwords
import csv

def load_imdb(path = "rawData\\imdb", bag_of_words_size = 25):
    file_path = path + "\\movie_metadata.csv" 
    data_list = _parse_csv_to_dict(file_path)
    movie_labels, movie_tensor, bag_of_words = _get_movies(data_list, bag_of_words_size)
    
    dataset = HeteroData()
    dataset['movie'].num_nodes = len(movie_labels)    
    dataset['movie'].y = torch.tensor(list(movie_labels))
    dataset['movie'].x = movie_tensor
    dataset['movie'].xKeys = bag_of_words

    return dataset




def _get_movies(data_list, bag_of_words_size):
    movies = {}
    total_words = {}
    labels = []
    bag_of_words = []
    labelNames = ["Action", "Comedy", "Drama"]

    for data in data_list:
        genres = data["genres"].split("|")
        for label_idx, labelName in enumerate(labelNames):
            if labelName in genres:
                labels.append(label_idx)
                movies[data["movie_title"]] = len(labels)
                plots = data["plot_keywords"].split("|")  # Tokenize and convert to lowercase
                word_count = defaultdict(int)
                for plot in plots:
                    plot = plot.replace(" ", "_")
                    if plot == "":
                        continue
                    if plot in total_words:
                        total_words[plot] += 1
                    else:
                        total_words[plot] = 1
                    word_count[plot] += 1
                bag_of_words.append(dict(word_count))
                continue

    total_words = dict(sorted(total_words.items(), key=lambda item: item[1], reverse=True)[:bag_of_words_size])
    vocabulary = list(total_words.keys())

    matrix = []
    for item in bag_of_words:
        vector = [item.get(word, 0) for word in vocabulary]
        matrix.append(vector)
    
    # Convert the matrix to a PyTorch tensor
    tensor = torch.tensor(matrix, dtype=torch.float32)
    return labels, tensor, vocabulary


def _parse_csv_to_dict(file_path):
    """
    Parses a CSV file and returns a list of dictionaries where each dictionary
    represents a row from the CSV with keys as column headers.
    
    :param filename: Path to the CSV file
    :return: List of dictionaries representing the CSV data
    """
    data_list = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            # Create a CSV reader object using the file object
            reader = csv.DictReader(file)
            # Iterate over the rows in the CSV file
            for row in reader:
                # Each row is a dictionary with header as keys
                data_list.append(row)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return data_list

# Example usage
if __name__ == "__main__":
    result = load_idmb()
    print(result)