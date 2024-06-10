from collections import defaultdict
import re
import torch
from torch_geometric.data import HeteroData
import nltk
from nltk.corpus import stopwords

def load_dblp(path = "rawData\\dblp", bag_of_words_size = 10):
    author_ids, author_labels, author_id_dict = _get_authors(path)
    paper_tensor, paper_id_dict, bag_of_words = _get_papers(path, bag_of_words_size)
    paper_author_mappings = _get_author_paper_mappings(path, author_id_dict, paper_id_dict)
    conf_ids, conf_id_dict = _get_conference(path)
    paper_conference_mappings = _get_paper_conference_mappings(path, author_id_dict, conf_id_dict)

    dataset = HeteroData()
    dataset['author'].num_nodes = len(author_labels)    
    #dataset['author'].x = torch.tensor([5] * len(author_labels)).unsqueeze(1)
    dataset['author'].y = torch.tensor(list(author_labels))
    dataset['paper'].x = paper_tensor
    dataset['paper'].xKeys = bag_of_words
    dataset['author', 'writes', 'paper'].edge_index = paper_author_mappings.t()
    dataset['conference'].num_nodes = len(conf_ids)
    dataset['paper', 'published_in', 'conference'].edge_index = paper_conference_mappings.t()
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

    #sorted_lists = sorted(zip(ids, labels))

    id_dict = {}
    for idx, id in enumerate(ids):
        id_dict[id] = idx
    # Unzip the sorted lists
    #ids, labels = zip(*sorted_lists)
    return ids, labels, id_dict

def _get_papers(path, bag_of_words_size):
    file_path = path + "\\paper.txt" 
    bag_of_words = []
    vocabulary = []
    total_words = {}
    paper_id_dict = {}
    id_paper_dict = {}
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
            id_paper_dict[idx] = id

    total_words = dict(sorted(total_words.items(), key=lambda item: item[1], reverse=True)[:bag_of_words_size])
    vocabulary = list(total_words.keys())

    matrix = []
    valid_paper_ids = []
    for idx, item in enumerate(bag_of_words):
        vector = [item.get(word, 0) for word in vocabulary]
        matrix.append(vector)
        #if any(vector):  # Check if vector is not all zeros
        #    matrix.append(vector)
        #    valid_paper_ids.append(idx)
    
    # Convert the matrix to a PyTorch tensor
    tensor = torch.tensor(matrix, dtype=torch.float32)
    
    # Filter out paper_id_dict to keep only valid papers
    #paper_id_dict = {}
    #for paper_id in valid_paper_ids:
    #    paper_id_dict = {id_paper_dict[paper_id]: paper_id }
    
    return tensor, paper_id_dict, vocabulary

def _get_author_paper_mappings(path, author_id_dict, paper_id_dict):
    file_path = path + "\\paper_author.txt"  # Replace with the path to your file

    mappings = []

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            paper, author = line.split("\t", 1)
            if paper in paper_id_dict and author in author_id_dict:
                mappings.append([author_id_dict[author],paper_id_dict[paper]])

    mappings = torch.tensor(mappings)
    return mappings

def _get_conference(path):
    file_path = path + "\\conf.txt" # Replace with the path to your file
    ids = []

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            id, name = line.split("\t")
            ids.append(id)

    #sorted_lists = sorted(zip(ids, labels))

    id_dict = {}
    for idx, id in enumerate(ids):
        id_dict[id] = idx
    # Unzip the sorted lists
    #ids, labels = zip(*sorted_lists)
    return ids, id_dict

def _get_paper_conference_mappings(path, paper_id_dict, conf_id_dict):
    file_path = path + "\\paper_conf.txt"  # Replace with the path to your file

    mappings = []

    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            paper, conference = line.split("\t", 1)
            if paper in paper_id_dict and conference in conf_id_dict:
                mappings.append([paper_id_dict[paper], conf_id_dict[conference]])

    mappings = torch.tensor(mappings)
    return mappings


if __name__ == "__main__":
    dataset = load_dblp(path = "rawData\\customDblp")
    print(dataset)
