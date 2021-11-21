from model import WGAN_GP
import torch
from utils.config import parse_args
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
  
    def __init__(self):
        'Initialization'
        
    def __len__(self):
        'Denotes the total number of samples'
        return 1200
        #return 6 #for testing code
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Load data
        X = torch.load('true_data/{}.pt'.format(index))
        return X

def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    dataset = Dataset()
    train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.2)
    train_data = Subset(dataset,train_index)
    valid_data = Subset(dataset,valid_index)
    train_loader = DataLoader(dataset = train_data, batch_size=60,
                                         shuffle=True)
    test_loader = DataLoader(dataset=valid_data, batch_size=60,shuffle=False)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)