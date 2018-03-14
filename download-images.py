"""
Download and preprocess images for the ImageGraph dataset.

Before running this script, please download the URL lists from
https://github.com/nle-ml/mmkb and unpack to data/fb15k-237/image-graph_urls.
Uses multiprocessing to process the images in parallel. Skips images that
were already downloaded.

Usage:
    python download-images.py [google|bing|yahoo]
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import urllib
import urlparse
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import errno
from ssl import CertificateError
from httplib import HTTPException
from multiprocessing import Pool
import sys


ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevent PIL from raising an error if a file is truncated

if len(sys.argv) > 1:
    provider = sys.argv[1]
else:
    provider = 'google'

urls = pd.read_csv('data/fb15k-237/image-graph_urls/URLS_{}.txt'.format(provider), sep='\t', names=['url', 'id'])


def download_image(i):

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
            pass
        else:
            try:
                # Download and open image.
                # TODO: Store temporary files in a separate folder (e.g. image-graph_temp),
                #       so they are not hidden in the system folders.
                temp_filename = ''
                temp_filename, _ = urllib.urlretrieve(row['url'])
                im = Image.open(temp_filename)
            except (IOError, CertificateError, HTTPException):
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
            finally:
                # Remove temporary file.
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass


pool = Pool(100)  # use many processes here because some of them will get stuck waiting for server responses

with tqdm(total=len(urls)) as pbar:
    for _ in pool.imap_unordered(download_image, urls.index, 100):
        pbar.update()
