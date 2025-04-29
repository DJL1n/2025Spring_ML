# utils/config.py

class EnConfig(object):
    def __init__(self,
                 learning_rate=1e-5,
                 dataset_name='mosi',
                 early_stop=8,
                 seed=0,
                 dropout=0.3,
                 batch_size=16,
                 num_hidden_layers=1,
                 use_context = False,
                 use_attnFusion=False,
                 ):
        self.train_mode = 'regression'
        self.loss_weights = {'M': 1.5, 'T': 1, 'A': 1}
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.model_save_path = 'checkpoint/'
        self.tasks = 'MTA'
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.use_context = use_context
        self.use_attnFusion = use_attnFusion

        self.nheads = 16
        self.dimension = 1024

        # 需要调试的参数
        self.seed = seed
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers

        self.text_context_length = 5
        self.audio_context_length = 3

