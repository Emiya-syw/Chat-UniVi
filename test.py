import torch

def merge_max(image_features, scores):
    print(image_features.shape[0], scores.shape[0])
    max_value, max_index = torch.max(scores, dim=0)

    image_features_new = torch.zeros((image_features.shape[0]-1, image_features.shape[1], image_features.shape[2])).to(image_features.device)
    image_features_new[:max_index] = image_features[:max_index]
    image_features_new[max_index] = 0.5*(image_features[max_index]+image_features[max_index+1])

    scores_new = torch.zeros((scores.shape[0]-1)).to(scores.device)
    scores_new[:max_index] = scores[:max_index]

    if max_index != image_features.shape[0]-2:
        image_features_new[max_index+1:] = image_features[max_index+2:]
        scores_new[max_index:] = scores[max_index+1:]
    else:
        pass
    
    return image_features_new, scores_new

def merge(image_features, num_merge):
    image_features_1 = image_features[:-1]
    image_features_2 = image_features[1:]
    scores = torch.mean(torch.sum(image_features_1 * image_features_2, dim=1), dim=-1)
    for i in range(num_merge):
        image_features, scores = merge_max(image_features, scores)
    return image_features
    

features = torch.rand((1024,256,1024))
while features.shape[0] > 128:
    print(features.shape)
    features = merge(features, 3)