

if __name__ == "__main__":
    fea = {}
    with open('sample_feature_test_20170814') as f:
        for i, line in enumerate(f.readlines()):
            ds = line.strip().split(" ")
            for f in ds[1:]:

                key, value = f.split(':')
                if not (int(key) == 1 and float(value) == 0.0):
                    key = int(key)
                    v = fea.get(key,0)
                    fea[key] = v + 1
    sta = {}
    for key in fea:
        value = fea[key]
        if value in sta:
            sta[value].append(key)
        else:
            sta[value] = [key]

    labels = []
    features = []
    with open('sample_feature_test_20170814') as f:
        print("-------read file-------")
        pos = 0
        for i, line in enumerate(f.readlines()):
            ds = line.strip().split(" ")
            label = int(float(ds[0]))
            if label == 0:
                label = -1
            feature = {}

            if label == 1:
                pos +=1

            for fea in ds[1:]:
                key, value = fea.split(':')
                feature[int(key)] = float(value)
            if not(1 in feature and feature[1] == 0.0):
                labels.append(label)
                features.append(feature)
    new_labels = []
    new_features = []
