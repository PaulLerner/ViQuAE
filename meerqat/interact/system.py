# -*- coding: utf-8 -*-

from jsonargparse import CLI

import ranx

from ..data.loading import IMAGE_PATH, get_pretrained
from ..image.embedding import get_model_and_transform as get_image_model, embed as embed_image
from ..ir.embedding import embed as embed_text
from ..ir.search import Searcher
from ..ir.fuse import Fusion


class System:
    """Interact with the KVQAE system"""
    def __init__(self, searcher_kwargs: dict, image_kwargs: dict, text_kwargs: dict, 
                 tokenizer_kwargs: dict, tokenization_kwargs: dict):
        self.searcher = Searcher(**searcher_kwargs)
        self.image_model = get_image_model(**image_kwargs)
        self.text_model = get_pretrained(**text_kwargs)
        self.tokenizer = get_pretrained(**tokenizer_kwargs)
        self.tokenization_kwargs = tokenization_kwargs       
        
    def pipeline(self, batch):
        # 1. process input
        batch = embed_image(batch, **self.image_model)
        batch = embed_text(batch, **self.text_model, **self.tokenizer, 
                           tokenization_kwargs=self.tokenization_kwargs)
        
        # 2. IR
        batch = self.searcher(batch)
        self.searcher.qrels = ranx.Qrels(self.searcher.qrels)
        for name, run in self.searcher.runs.items():
            self.searcher.runs[name] = ranx.Run(run, name=name)      
        fuser = Fusion(
            qrels=self.searcher.qrels,
            runs=list(self.searcher.runs.values()),    
            **self.searcher.fusion_kwargs
        )
        run = fuser.test(**self.searcher.fusion_kwargs['subcommand_kwargs'])
        
        # 3. RC: TODO
        
    def user_loop(self):
        image = None
        while True:
            # TODO download from URL
            # 1. image
            if image is None:
                answer = input(f"Enter the image file name stored in '{IMAGE_PATH}' or enter 'q' to quit.\n")
            else:
                answer = input(f"Enter the image file name stored in '{IMAGE_PATH}' or press Enter to keep the previous one or enter 'q' to quit.\n")   
            answer = answer.strip()
            if answer.lower() == 'q':
                break
            elif len(answer) > 0:
                image = answer
            # else keep previous image
            
            # 2. question
            answer = input("Ask your question in English or enter 'q' to quit.\n") 
            answer = answer.strip()
            if answer.lower() == 'q':
                break
            question = answer                       
            
            # 3. answer
            batch = {'image': [image], 'input': [question], 'id': ['FAKE'], 'output': ['FAKE']}
            self.pipeline(batch)
            print(f"> {answer}\n")
        
    
if __name__ == '__main__':
    CLI(System)