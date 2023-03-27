import datasets
import csv
from glob import glob
from tempfile import NamedTemporaryFile
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import os


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
        


float_variables = 'n_tokens_title,n_tokens_content,n_unique_tokens,n_non_stop_unique_tokens,num_hrefs,num_self_hrefs,num_imgs,num_videos,average_token_length,num_keywords,data_channel_is_lifestyle,data_channel_is_entertainment,data_channel_is_bus,data_channel_is_socmed,data_channel_is_tech,data_channel_is_world,kw_min_min,kw_max_min,kw_avg_min,kw_min_max,kw_max_max,kw_avg_max,kw_min_avg,kw_max_avg,kw_avg_avg,self_reference_min_shares,self_reference_max_shares,self_reference_avg_sharess,weekday_is_monday,weekday_is_tuesday,weekday_is_wednesday,weekday_is_thursday,weekday_is_friday,weekday_is_saturday,weekday_is_sunday,LDA_00,LDA_01,LDA_02,LDA_03,LDA_04,global_subjectivity,global_sentiment_polarity,global_rate_positive_words,global_rate_negative_words,rate_positive_words,rate_negative_words,avg_positive_polarity,min_positive_polarity,max_positive_polarity,avg_negative_polarity,min_negative_polarity,max_negative_polarity,title_subjectivity,title_sentiment_polarity,abs_title_subjectivity,abs_title_sentiment_polarity'.split(',')
        
        
class online_news_popularity(datasets.GeneratorBasedBuilder):
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [online_news_popularity_Config()]

#     DEFAULT_CONFIG_NAME = 'online news popularity data'
    

    def _info(self):
        main_features = {
                "title": datasets.Value("string"),
                "content": datasets.Value("string"),
                "shares" : datasets.Value("int32"),
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
        downloaded_files = dl_manager.download({'train_path':_FILEPATH, 
                                                'valid_path':_FILEPATH})

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split_key": "train", 'filepath':downloaded_files['train_path']},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split_key": "validate", 'filepath':downloaded_files['valid_path']},
            ),
        ]            
            
            
    def _get_examples_from_split(self, split_key, filepath):
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(filepath)
        df = df.loc[df.notnull().prod(axis = 1).astype(bool),:].reset_index(drop = True)
        df_train, df_test = train_test_split(df, test_size = .2, random_state = 2023)
        
        if split_key.lower() == "train":
            return df_train
        elif split_key.lower() == "validate":
            return df_test
        else:
            raise ValueError(f"Invalid split key {split_key}") 
            
    def _generate_examples(self, split_key, filepath):
        df = self._get_examples_from_split(split_key, filepath) #, file_path)
        for i, row in df.iterrows():
            yield i, row.to_dict()
        
#         for i, t, c, s in zip(range(len(title)), title, content, shares):
#             yield i, {'title':t, 'content':c, 'shares':s}
                
