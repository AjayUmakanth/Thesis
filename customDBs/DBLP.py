from collections import defaultdict
import re
import torch
from torch_geometric.data import HeteroData
import nltk
from nltk.corpus import stopwords

def load_dblp(path = "rawData\dblp"):
    author_ids, author_labels, author_id_dict = _get_authors(path)
    paper_tensor, paper_id_dict = _get_papers(path)
    paper_author_mappings = _get_author_paper_mappings(path, author_id_dict, paper_id_dict)

    dataset = HeteroData()
    dataset['author'].x = torch.tensor([0] * len(author_labels)).unsqueeze(1)
    dataset['author'].y = torch.tensor(list(author_labels))
    dataset['paper'].x = paper_tensor
    dataset['author', 'writes', 'paper'].edge_index = paper_author_mappings.t()
    return dataset

def _get_authors(path):
    file_path = path + "\\author_label.txt" # Replace with the path to your file
    ids = []
    labels = []

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            id, label, name = line.split("\t")
            ids.append(id)
            labels.append(int(label))

    sorted_lists = sorted(zip(ids, labels))

    id_dict = {}
    for idx, id in enumerate(ids):
        id_dict[id] = idx
    # Unzip the sorted lists
    ids, labels = zip(*sorted_lists)
    return ids, labels, id_dict

def _get_papers(path):
    file_path = path + "\paper.txt" 
    bag_of_words = []

    # Vocabulary to store unique words
    vocabulary = []
    ids = []
    total_words = {}
    paper_id_dict = {}
    stop_words = stopwords.words('english')

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            id, text = line.split("\t", 1)
            words = re.findall(r'\w+', text.lower())  # Tokenize and convert to lowercase
            word_count = defaultdict(int)
            for word in words:
                if word in stop_words:
                    continue
                if word not in vocabulary:
                    vocabulary.append(word)
                if word in total_words:
                    total_words[word] += 1
                else:
                    total_words[word] = 1
                word_count[word] += 1
            bag_of_words.append(dict(word_count))
            paper_id_dict[id] = idx

    total_words = dict(sorted(total_words.items(), key=lambda item: item[1], reverse=True))

    matrix = []
    for item in bag_of_words:
        vector = [item.get(word, 0) for word in vocabulary]
        matrix.append(vector)

    # Convert the matrix to a PyTorch tensor
    tensor = torch.tensor(matrix, dtype=torch.float32)

    return tensor, paper_id_dict

def _get_author_paper_mappings(path, author_id_dict, paper_id_dict):
    file_path = path + "\paper_author.txt"  # Replace with the path to your file

    mappings = []

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            paper,author = line.split("\t", 1)
            if paper in paper_id_dict and author in author_id_dict:
                mappings.append([paper_id_dict[paper],author_id_dict[author]])

    return torch.tensor(mappings)


if __name__ == "__main__":
    dataset = load_dblp()
    print(dataset)