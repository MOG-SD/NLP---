import Vocab
import logging
import Data_processing
import model_define

train_data, dev_data, test_data = Data_processing.split_data(10,1)
vocab = Vocab.Vocab(train_data)
model = model_define.Model(vocab)

# train
trainer = model_define.Trainer(model, vocab)
trainer.train()

logging.info("Training finished.")

# test
trainer.test()

logging.info("Training finished.")