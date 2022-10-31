# -*- coding: utf-8 -*-
from torch import nn

from transformers import BertModel
from transformers.models.bert import BertPreTrainedModel

from .mm import ECAEncoder
from .outputs import ReRankerOutput


class BertReRanker(BertPreTrainedModel):
    """
    As described in [1]_.
    
    Almost like BertForSequenceClassification without dropout, and pooling from [CLS] token.
    
    References
    ----------
    .. [1] Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 
       2019. Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering. 
       In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing 
       and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 
       pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, *args, **kwargs):
        outputs = self.bert(*args, **kwargs)
        
        # Pool from [CLS]
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        
        return ReRankerOutput(
            logits=logits, 
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class ECAReRanker(ECAEncoder):
    """Like BertReRanker with a ECA backbone instead of BERT"""
    def __init__(self, config):
        super().__init__(config)        
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, *args, return_dict=True, **kwargs):
        outputs = super().forward(*args, return_dict=return_dict, **kwargs)        
        logits = self.classifier(outputs.pooler_output)        
        return ReRankerOutput(
            logits=logits, 
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

        

