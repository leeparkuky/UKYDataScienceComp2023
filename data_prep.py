import datasets

#async packages
from aiohttp import ClientSession
import asyncio
# basic python packages
import requests
import re
# dataframes
import pandas as pd
# file systems
from glob import glob
import os
import sys
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import csv


summary_df = pd.read_csv('text_summary_stats.csv')



def gen_temp_file():
    if len(glob('text_datasets/*.csv')):
        pass
    else:
        try:
            subprocess.run(['python','data_prep.py'])
        except:
            subprocess.run(['python3','data_prep.py'])

    csv_files = glob('text_datasets/*.csv')
    
    
    if os.path.exists(os.path.join(os.getcwd(), 'online_news_popularity_data')):
        pass
    else:
        os.mkdir(os.path.join(os.getcwd(), 'online_news_popularity_data'))
                      
    
    fpath = os.path.join(os.getcwd(), 'online_news_popularity_data', 'online_news_popularity_data.csv')
#     with NamedTemporaryFile(mode='w', suffix = '.csv', 
#                             dir = os.path.join(os.getcwd(), 'online_news_popularity_data'), 
#                             delete=False) as f:


    with open(fpath, 'w') as f:
        fieldnames = pd.read_csv(csv_files[0]).columns.tolist()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for csv_file in tqdm(csv_files):
            with open(csv_file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row_dict = {fieldname: row[fieldname] for fieldname in fieldnames}
                    writer.writerow(row_dict)
#                     writer.writerow({'title': row['title'],
#                                     'content': row['content'],
#                                     'shares': row['shares']})
    return fpath








def get_texts(urls, shares):
    df = asyncio.run(download_all_data(urls, shares))
    return df


async def text_download(url, session):
    async with session.get(url) as resp:
        if resp.status == 200:
            try:
                html = await resp.read()
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title').text.split(' | ')[0]
                paragraphs = soup.find_all('p')
                texts = [p.text for p in paragraphs if 'nggallery' not in p.text]
                text = '\n'.join(texts)
            except:
                title, text = None, None
            return title, text
        else:
            return None, None
        

async def download_all_data(urls, shares):
    async with ClientSession() as session:
        tasks = [text_download(url, session) for url in urls]
        full_texts = await asyncio.gather(*tasks)
        titles = [text[0] for text in full_texts]
        contents = [text[1] for text in full_texts]
        df = pd.DataFrame(zip(titles, contents, shares), columns = ['title','content','shares'])
        return df

def save_text_csv(urls, shares, file_name, summary):
    try:
        dataset_dir = os.path.join(os.getcwd(), 'text_datasets')
        if os.path.exists(dataset_dir):
            pass
        else:
            os.mkdir(dataset_dir)

        df = asyncio.run(download_all_data(urls, shares))
        df = pd.concat([df.reset_index(drop = True), summary.reset_index(drop = True)], axis = 1)
        df.to_csv(os.path.join(dataset_dir, file_name), index = False)
    except:
        pass
    return None






if __name__ == '__main__':
    
    dl_manager = datasets.DownloadManager()
    _DOWNLOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
    archive = dl_manager.download(_DOWNLOAD_URL)

    for path, f in dl_manager.iter_archive(archive):
        if path[-3:] == 'csv':
            df = pd.read_csv(f)

    df.columns = df.columns.str.strip()
    urls = df.url.str.replace('http://', 'https://')
    shares = df.shares
    N = df.shape[0]
    batch_size = 100
    print('start process')
    res = Parallel(n_jobs = -1)(delayed(save_text_csv)(urls[i*batch_size:(i+1)*batch_size], shares[i*batch_size:(i+1)*batch_size], f"dataset_{i}.csv", summary_df.loc[i*batch_size:(i+1)*batch_size,:]) for i in tqdm(range(N//batch_size+1)))
    
    path = gen_temp_file()
    df = pd.read_csv(path)
    df = df.loc[df.notnull().prod(axis = 1).astype(bool),:].reset_index(drop = True)
    df.to_csv(path, index = False)

#     text_df = get_texts(urls[:1000], shares[:1000])
#     save_text_csv(urls[:100], shares[:100], 'text_data.csv')
#     text_df.to_csv('text_data.csv', index = False)

# res = Parallel(n_jobs=-1)(delayed(fun)() for fun in self.functions.values())