import datasets
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import os
import csv
from csv import DictReader

_DESCRIPTION = """\
This data is used for the data science competition at University of Kentucky in 2023
"""

_CITATION = """\
@InProceedings{10.1007/978-3-319-23485-4_53,
author="Fernandes, Kelwin
and Vinagre, Pedro
and Cortez, Paulo",
editor="Pereira, Francisco
and Machado, Penousal
and Costa, Ernesto
and Cardoso, Am{\'i}lcar",
title="A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News",
booktitle="Progress in Artificial Intelligence",
year="2015",
publisher="Springer International Publishing",
address="Cham",
pages="535--546",
abstract="Due to the Web expansion, the prediction of online news popularity is becoming a trendy research topic. In this paper, we propose a novel and proactive Intelligent Decision Support System (IDSS) that analyzes articles prior to their publication. Using a broad set of extracted features (e.g., keywords, digital media content, earlier popularity of news referenced in the article) the IDSS first predicts if an article will become popular. Then, it optimizes a subset of the articles features that can more easily be changed by authors, searching for an enhancement of the predicted popularity probability. Using a large and recently collected dataset, with 39,000 articles from the Mashable website, we performed a robust rolling windows evaluation of five state of the art models. The best result was provided by a Random Forest with a discrimination power of 73{\%}. Moreover, several stochastic hill climbing local searches were explored. When optimizing 1000 articles, the best optimization method obtained a mean gain improvement of 15 percentage points in terms of the estimated popularity probability. These results attest the proposed IDSS as a valuable tool for online news authors.",
isbn="978-3-319-23485-4"
}
"""

_FILEPATH = os.path.join(os.getcwd(),'online_news_popularity_data', 'online_news_popularity_data.csv')
    
with open(_FILEPATH, 'r') as f:
    reader = DictReader(f)
    float_variables = [x for x in reader.fieldnames if x not in ['title','content','shares','shares_class']]
    
# _FILEPATH = os.path.join(os.getcwd(), 'online_news_popularity_data.csv')

    
    
    
class online_news_popularity_Config(datasets.BuilderConfig):
    """BuilderConfig for sample_data."""

    def __init__(self, **kwargs):
        super().__init__(
            description=f"raw text data for articles used in Fernandes et. al.'s paper in the citation",
            version=datasets.Version("1.0.0",""),
            **kwargs,
        )
        self.name = 'online_news_popularity_data'
        self.date = "20230322"
        self.language = "en"
        


        
class online_news_popularity(datasets.GeneratorBasedBuilder):
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [online_news_popularity_Config()]

#     DEFAULT_CONFIG_NAME = 'online news popularity data'
    

    def _info(self):
        main_features = {
                "title": datasets.Value("string"),
                "content": datasets.Value("string"),
                "shares" : datasets.Value("float"),
                }
        main_features.update({var: datasets.Value("float") for var in float_variables})
        main_features.update({'shares_class': datasets.features.ClassLabel(num_classes=2, names=["neg", "pos"])})
        
        return datasets.DatasetInfo(
            description="UKY Data Science Competition 2023 Raw Data",
            features=datasets.Features(
                main_features
            ),
            # No default supervised_keys.
            supervised_keys=None,
            homepage=None,
            citation= _CITATION
        )
    
    
          
    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(_FILEPATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath':downloaded_files},  #downloaded_files['train_path']
            ),
        ]            
            
            

    def _generate_examples(self, filepath):
        with open(filepath, 'r') as f:
            reader = DictReader(f)
            for i, row in enumerate(reader):
                yield i, row
