import os, json, numpy as np
from tqdm import tqdm
from config import *

def load_sequence(folder):
    frames = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".json"): continue
        data = json.load(open(os.path.join(folder, f)))
        hc3d = np.array(data["hc3d"], dtype=np.float32)
        frames.append(hc3d)
    return frames

def pad_or_crop(frames):
    if len(frames) >= SEQ_LEN:
        return np.array(frames[:SEQ_LEN])
    else:
        pad_len = SEQ_LEN - len(frames)
        pad = np.zeros((pad_len, NUM_JOINTS, INPUT_FEATURES))
        return np.concatenate([frames, pad], axis=0)

def main():
    os.makedirs(OUTPUT_DATASET, exist_ok=True)
    X, y = [], []
    label_map = {}

    people = sorted(os.listdir(DATASET_ROOT))
    label_id = 0

    for person in tqdm(people):
        person_path = os.path.join(DATASET_ROOT, person)
        if not os.path.isdir(person_path): continue

        label_map[person] = label_id

        for seq in os.listdir(person_path):
            seq_path = os.path.join(person_path, seq)
            frames = load_sequence(seq_path)

            if len(frames) == 0: continue

            seq_padded = pad_or_crop(frames)
            X.append(seq_padded)
            y.append(label_id)

        label_id += 1

    np.save(f"{OUTPUT_DATASET}/X.npy", np.array(X))
    np.save(f"{OUTPUT_DATASET}/y.npy", np.array(y))
    json.dump(label_map, open(f"{OUTPUT_DATASET}/labels.json", "w"))

    print("Done:", len(X), "samples")

if __name__ == "__main__":
    main()
