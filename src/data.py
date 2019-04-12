from torch.utils.data import Dataset

from nltk.corpus import framenet as fn


class FrameNetDataset(Dataset):

   def __init__(self):
       self.sentences = fn.sents(exemplars=True, full_text=False)


   def __len__(self):
      return 171839

   def __getitem__(self, idx):
       input  = self.sentences[idx].text
       label = self.sentences[idx].frame.ID
       return (input, label)


