# Download all images from the URL lists in parallel.

from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

import urllib
import urlparse
from PIL import Image
from tqdm import tqdm
import os
import errno
from ssl import CertificateError
from httplib import HTTPException
from multiprocessing import Pool


provider = 'google'
urls = pd.read_csv('data/fb15k-237/image-graph_urls/URLS_{}.txt'.format(provider), sep='\t', names=['url', 'id'])


def download_image(i):

    #global counter_fail, counter_success, counter_skip

    row = urls.iloc[i]
    url = row['url']
    freebase_id, index = row['id'].split('/')
    index = int(index)

    # Create dir for entity if it doesn't exist.
    target_dir = 'data/fb15k-237/image-graph_images/{}'.format(freebase_id)
    try:
        os.makedirs(target_dir)
    except OSError as e:  # multiprocessing-safe way to handle existing dir
        if e.errno != errno.EEXIST:
            raise

    # TODO: Maybe use more images if some cannot be downloaded.
    if index < 25:  # only use first 25 images (as in the paper)
        target_filename = '{}_{}.jpg'.format(provider, index)

        # Skip images that were already downloaded.
        if os.path.exists(os.path.join(target_dir, target_filename)):
            #counter_skip += 1
            pass
        else:
            try:
                # Download and open image.
                temp_filename, _ = urllib.urlretrieve(row['url'])
                im = Image.open(temp_filename)
            except (IOError, CertificateError, HTTPException):
                #print('Failed:', row['url'])
                #counter_fail += 1
                pass
            except Exception as e:
                print('Got unusual error during downloading/opening of image:', e)
                print('Please make sure that this error is just caused by a corrupted file.')
            else:
                # Resize and convert to jpg.
                im.thumbnail((500, 500), Image.ANTIALIAS)
                im = im.convert('RGB')

                # Save.
                im.save(os.path.join(target_dir, target_filename))
                #counter_success += 1


#counter_fail = 0
#counter_success = 0
#counter_skip = 0

pool = Pool(100)  # use many processes here because some of them will get stuck waiting for server responses

with tqdm(total=len(urls)) as pbar:
    for _ in pool.imap_unordered(download_image, urls.index, 100):
        pbar.update()

#print('Successfully downloaded', counter_success, 'images, failed on', counter_fail, 'images, skipped', counter_skip, 'images')
