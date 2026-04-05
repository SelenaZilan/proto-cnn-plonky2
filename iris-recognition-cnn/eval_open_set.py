# Open-set evaluation pipeline references https://github.com/AndrejHafner/iris-recognition-cnn

import json
import pathlib
from collections import Counter

import torch
import numpy as np

from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallIrisCNN


def _load_checkpoint_payload(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("model_config", {})
    return payload, {}


def get_model(model_name, checkpoint_path, num_classes=1500):

    model = None
    input_size = 0
    state_dict, model_config = _load_checkpoint_payload(checkpoint_path)

    if model_name == "smalliris":
        model = SmallIrisCNN(
            num_classes=num_classes,
            embedding_dim=model_config.get("embedding_dim", 128),
            c1=model_config.get("c1", 32),
            c2=model_config.get("c2", 64),
        )
        input_size = model_config.get("input_size", 64)
        model.load_state_dict(state_dict)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

def get_dataloader(data_path, input_size, batch_size=32):

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def enroll_identities(feature_extract_func, dataloader, device):
    enrolled = {}
    with torch.no_grad():
        for input, labels in dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()

            predictions = feature_extract_func(inputs).cpu().detach().numpy()

            unique_labels = np.unique(labels)
            for i in unique_labels:
                user_features = predictions[labels == i, :]
                if i in enrolled:
                    enrolled[i] = np.vstack((enrolled[i], normalize(user_features, axis=1, norm='l2')))
                else:
                    enrolled[i] = normalize(user_features, axis=1, norm='l2')

    return enrolled

def evaluate(enrolled, feature_extract_func, dataloader, device, rank_n=50):
    total = 0
    rank_n_correct = np.zeros(rank_n)

    with torch.no_grad():
        for input, labels in dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()
            predictions = feature_extract_func(inputs).cpu().detach().numpy()
            for idx, label in enumerate(labels):
                pred = predictions[idx, :].reshape(-1, 1)
                pred_norm = normalize(pred, axis=0, norm="l2")
                similarities_id = {}
                for key in enrolled.keys():
                    cosine_similarities = np.matmul(enrolled[key], pred_norm)
                    similarities_id[key] = np.max(cosine_similarities)

                counter = Counter(similarities_id)
                for i in range(1, rank_n + 1):
                    rank_n_vals = list(dict(counter.most_common(i)).keys())
                    rank_n_correct[i-1] += 1 if label in rank_n_vals else 0
                total +=1

    rank_n_correct /= total
    rank_1_accuracy =  rank_n_correct[0]
    rank_5_accuracy = rank_n_correct[4]
    print(f"Rank 1 accuracy: {rank_1_accuracy}, rank 5 accuracy: {rank_5_accuracy}")
    return rank_1_accuracy, rank_5_accuracy, rank_n_correct

if __name__ == '__main__':

    print("Loading model...")
    checkpoint_path = "./models/smalliris_e_40_lr_0_001_in_48_c1_24_c2_48_emb_64_best.pth"
    model_name = "smalliris"

    enrollment_data_path = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment"
    test_data_path = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test"
    batch_size = 196

    model, input_size = get_model(model_name, checkpoint_path)

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    enrollment_dataloader = get_dataloader(enrollment_data_path, input_size, batch_size=batch_size)
    test_dataloader = get_dataloader(test_data_path, input_size, batch_size=batch_size)

    print("Enrolling identities...")
    enrolled = enroll_identities(model.feature_extract_avg_pool, enrollment_dataloader, device)

    print("Running recognition evaluation...")
    rank_1_accuracy, rank_5_accuracy, rank_n_accuracy = evaluate(enrolled, model.feature_extract_avg_pool, test_dataloader, device)

    results = {
        "rank_1_acc": rank_1_accuracy,
        "rank_5_acc": rank_5_accuracy,
        "rank_n_accuracies": list(rank_n_accuracy)
    }

    pathlib.Path("./results").mkdir(parents=True, exist_ok=True)

    with open(f'./results/{model_name}_open_set_48_c1_24_c2_48_emb_64.json', 'w') as f:
        json.dump(results, f)
