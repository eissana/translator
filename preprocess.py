import spacy
from torchtext.data import Field
from torchtext.datasets import Multi30k


class Preprocessor:
    INIT_TOKEN = "<init>"
    EOS_TOKEN = "<eos>"  # end of sentence

    def __init__(self, spacy_names, exts, data_root, min_freq, max_size):
        self.src_spacy = spacy.load(spacy_names[0])
        self.tgt_spacy = spacy.load(spacy_names[1])

        def src_tokenize(text):
            return [t.text for t in self.src_spacy.tokenizer(text)]

        def tgt_tokenize(text):
            return [t.text for t in self.tgt_spacy.tokenizer(text)]

        self.src_field = Field(
            tokenize=src_tokenize,
            init_token=self.INIT_TOKEN,
            eos_token=self.EOS_TOKEN,
            lower=True,
        )
        self.tgt_field = Field(
            tokenize=tgt_tokenize,
            init_token=self.INIT_TOKEN,
            eos_token=self.EOS_TOKEN,
            lower=True,
        )

        # Run the following command to download data:
        # > sh multi30k.sh
        self.train, self.val, self.test = Multi30k.splits(
            exts=exts,
            fields=(self.src_field, self.tgt_field),
            root=data_root,  # data/multi30k/
        )
        self.src_field.build_vocab(self.train, max_size=max_size, min_freq=min_freq)
        self.tgt_field.build_vocab(self.train, max_size=max_size, min_freq=min_freq)
