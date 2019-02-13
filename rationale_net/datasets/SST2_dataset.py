import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
import pdb

SMALL_TRAIN_SIZE = 100

@RegisterDataset('SST2')
class StanfordSentiment(AbstractDataset):

    def __init__(self, args, word_to_indx, mode, max_length=90, stem='raw_data/stanford-sentiment-dataset-master/stsa.binary.'):
        self.args= args
        self.name = mode
        self.objective = args.objective
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}
        with open(stem+self.name) as f:
            lines = f.readlines()
            lines = list(zip( range(len(lines)), lines) )
            if args.debug_mode:
                lines = lines[:SMALL_TRAIN_SIZE]
            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = self.processLine(line_content, indx)

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                self.dataset.append(sample)
            f.close()
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("SST2 dataset doesn't support balanced sampling!")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line, i):
        if isinstance(line, bytes):
            line = line.decode()
        label = float(line[0])
        if self.objective == 'mse':
            raise NotImplementedError("SST2 dataset only allows binary classification")
        else:
            self.args.num_class = 2
        text_list = line[2:].split()[:self.max_length]
        text = " ".join(text_list)
        x =  get_indices_tensor(text_list, self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'i':i}
        return sample
