import os
import zipfile

from six.moves import urllib


def download_and_uncompress_zip(zip_url, dataset_dir):
    """Downloads the `zip_url` and uncompresses it locally.
       From: https://github.com/tensorflow/models/blob/master/slim/datasets/dataset_utils.py

    Args:
      zip_url: The URL of a zip file.
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = zip_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if tf.gfile.Exists(filepath):
        print('Zip file already exist. Skip download..', filepath)
    else:
        filepath, _ = urllib.request.urlretrieve(zip_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    with zipfile.ZipFile(filepath) as f:
        print('Extracting ', filepath)
        f.extractall(dataset_dir)
        print('Successfully extracted')
