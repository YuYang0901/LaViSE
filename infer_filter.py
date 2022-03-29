import cv2
import time
import argparse
from image_datasets import *
from model_loader import setup_explainer
from torchtext.vocab import GloVe


def inference(args):
    f = args.f
    method = args.method
    num_top_samples = args.p

    # prepare the pretrained word embedding vectors
    embedding_glove = GloVe(name='6B', dim=args.word_embedding_dim)
    embeddings = embedding_glove.vectors.T.cuda()

    # prepare the reference dataset
    if args.refer == 'vg':
        dataset = VisualGenome(transform=data_transforms['val'])
    elif args.refer == 'coco':
        dataset = MyCocoDetection(root='./data/coco/val2017',
                                  annFile='./data/coco/annotations/instances_val2017.json',
                                  transform=data_transforms['val'])
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # load the target model with a trained explainer
    model = setup_explainer(args, random_feature=args.random)
    ckpt = torch.load(args.model_path)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # get the max activation of each examples on the target filter
    max_activations = np.zeros(len(dataset))
    if not args.max_path:
        print('extracting max activations...')
        for k, batch in enumerate(dataloader):
            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name is args.layer:
                    break
            x = x.cpu().detach().numpy()
            max_activations[k] = np.max(x, axis=(-1, -2))[f]
        torch.save(max_activations, args.max_path)
        print('activations of filters saved!')
    max_activations = torch.load(args.max_path)

    # sort images by their max activations
    sorted_samples = np.argsort(-max_activations, axis=1)

    # load the activation threshold
    threshold = np.load(args.thresh_path)[f]

    with torch.no_grad():
        start = time.time()
        print('explaining filter %d with %d top activated images' % (f, num_top_samples))
        filter_dataset = torch.utils.data.Subset(dataset, sorted_samples[f, :num_top_samples])
        filter_dataloader = torch.utils.data.DataLoader(filter_dataset, batch_size=1,
                                                        shuffle=False, num_workers=0)
        weights = 0
        for i, batch in enumerate(filter_dataloader):
            if not batch[1]:
                continue
            data_, annotation = batch[0].cuda(), batch[1]
            x = data_.clone()
            for name, module in model._modules.items():
                x = module(x)
                if name is args.layer:
                    activation = x.detach().cpu().numpy()
                    break
            c = activation[:, f, :, :]
            c = c.reshape(7, 7)
            xf = cv2.resize(c, (224, 224))
            weight = np.amax(c)
            if weight <= 0.:
                continue

            # interpret the explainer's output with the specified method
            predict = explain(model, data_, method, activation, c, xf, threshold)
            predict_score = torch.mm(predict, embeddings) / \
                            torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                     torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
            sorted_predict_score, sorted_predict = torch.sort(predict_score, dim=1, descending=True)
            sorted_predict = sorted_predict[0, :].detach().cpu().numpy()
            select_rank = np.repeat(sorted_predict[:args.s], int(weight))

            if weights == 0:
                filter_rank = select_rank
            else:
                filter_rank = np.concatenate((filter_rank, select_rank))

            weights += weight

        with open('data/entities.txt') as file:
            all_labels = [line.rstrip() for line in file]
        (values, counts) = np.unique(filter_rank, return_counts=True)
        ind = np.argsort(-counts)
        sorted_predict_words = []
        for ii in ind[:args.num_output]:
            word = embedding_glove.itos[int(values[ii])]
            if word in all_labels:
                sorted_predict_words.append(word)

        end = time.time()
        print('Elasped Time: %f s' % (end - start))

    return sorted_predict_words


def explain(method, model, data_, activation, c, xf, threshold):
    img = data_.cpu().detach().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    if method == 'original':
        # original image
        data = data_.clone().requires_grad_(True)
        predict = model(data)
    elif method == 'projection':
        # filter attention projection
        filter_embed = torch.tensor(
            np.mean(activation * c / (np.sum(c ** 2, axis=(0, 1)) ** .5), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    elif method == 'image':
        # image masking
        data = img * (xf[:, :, None] > threshold)
        data = torch.tensor(np.transpose(data, (2, 0, 1))).unsqueeze(0).cuda()
        predict = model(data)
    elif method == 'activation':
        # activation masking
        filter_embed = torch.tensor(np.mean(activation * (c > threshold), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    else:
        raise NotImplementedError

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    parser.add_argument('--f', type=int, help='index of the target filter')
    parser.add_argument('--method', type=str, default='projection',
                        choices=('original', 'image', 'activation', 'projection'),
                        help='method used to explain the target filter')
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--refer', type=str, default='vg', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--num-output', type=int, default=10,
                        help='number of words used to explain the target filter')
    parser.add_argument('--model-path', type=str, default='', help='path to load the target model')
    parser.add_argument('--thresh-path', type=str, help='path to save/load the thresholds')
    parser.add_argument('--max-path', type=str, help='path to save/load the max activations of all examples')

    # if filter activation projection is used
    parser.add_argument('--s', type=int, default=5,
                        help='number of semantics contributed by each top activated image')
    parser.add_argument('--p', type=int, default=25,
                        help='number of top activated images used to explain each filter')

    args = parser.parse_args()
    print(args)

    inference(args)