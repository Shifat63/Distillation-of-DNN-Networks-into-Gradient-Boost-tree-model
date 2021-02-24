from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        '''
        Pytorch provides class Dataset, which we are extending here and
        customizing, with respect to our dataset. We are loading the csv
        file as dataframe, we are storing the inputs to X(attributes according
        to which we are making predictions), and y(original targets). We are
        scaling our input variables using StandardScaler(). Original targets are
        not scaled, as the target value is 0 or 1.

        Parameters
        ----------
        path : path to the heart.csv file

        Returns
        -------
        None.

        '''
        # load the csv file as a dataframe
        df = read_csv(path, header=None)

        # store the inputs and outputs
        self.X = df.iloc[1:, :-1].values
        self.y = df.iloc[1:, -1].values

        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')

        # fit scaler on data
        # apply transform
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        '''
        This function returns the number of rows in the dataset (number of samples
        we are working with).
        '''
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        '''
        This function returns the sample at the index idx
        '''
        return [self.X[idx], self.y[idx]]

    def get__inputs(self, idx):
        '''
        This function returns the sample, without original target, at the index idx.
        '''
        return self.X[idx]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        '''
        This function determines sizes for training and testing data, according
        to the n_test. Usually, 30%(0.3) of data is used for testing, 70% is used
        for training.

        Parameters
        ----------
        n_test : percentage of data used for testing(real number between 0 and 1)

        Returns
        -------
        indexes for train and test rows

        '''
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size

        # calculate the split
        return random_split(self, [train_size, test_size])

# prepare the dataset
def prepare_data(path):
    '''
    Pytorch provides DataLoader class. The aim of this class is to load
    Dataset instance during the model training and evaluation. Indexes
    for rows of data, which will be used for training and testing,
    returned by get_splits function, are passed to DataLoader, along with
    batch_size (we have opted to propagate one by one sample through the
    network) and shuffle parameter, which tells us whether the data should
    be shuffled every epoch. As that is better learning strategy, we have
    opted to shuffle the training dataset every epoch.

    Parameters
    ----------
    path : path to the heart.csv file

    Returns
    -------
    train_dl : training dataset
    test_dl : test dataset

    '''
    # load the dataset
    dataset = CSVDataset(path)

    # calculate split
    train, test = dataset.get_splits()

    # prepare data loaders
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)
    return train_dl, test_dl