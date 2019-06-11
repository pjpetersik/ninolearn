from __future__ import print_function

from ftplib import FTP
from requests import get
from os.path import isfile, join, exists
from os import remove, mkdir
import gzip
import shutil

from ninolearn.pathes import rawdir

if not exists(rawdir):
    print("make a data directory at %s" % rawdir)
    mkdir(rawdir)


def downloadFileFTP(info_dict, outdir='',
                    username='anonymous', password='anonymous_pass'):
    """
    Download a file from a FTP server
    """
    filename = info_dict['filename']
    host = info_dict['host']
    location = info_dict['location']
    if not exists(join(rawdir, outdir)):
        print("make a data directory at %s" % join(rawdir, outdir))
        mkdir(join(rawdir, outdir))

    if not isfile(join(rawdir, outdir, filename)):
        print("Download %s" % filename)
        ftp = FTP(host)
        ftp.login(username, password)
        ftp.cwd(location)

        localfile = open(join(rawdir, outdir, filename), 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write, 204800)
        localfile.close()
        ftp.quit()

    else:
        print("%s already downloaded" % filename)


def downloadFileHTTP(info_dict, outdir=''):
    """
    download a file via HTTP
    """
    # open in binary mode
    filename = info_dict['filename']
    url = info_dict['url']

    if not isfile(join(rawdir, outdir, filename.replace(".gz", ""))):
        print("Download %s" % filename)
        with open(join(rawdir, outdir, filename), "wb") as file:
            # get request
            response = get(url)

            # write to file
            file.write(response.content)
    else:
        print("%s already downloaded" % filename)


def unzip_gz(info_dict):
    """
    unzip .gz format files
    """
    filename_old = info_dict['filename']
    filename_new = filename_old.replace(".gz", "")

    # unpack file
    if not isfile(join(rawdir, filename_new)):
        print("Unzip %s" % filename_old)
        with gzip.open(join(rawdir, filename_old), 'rb') as f_in:
            with open(join(rawdir, filename_new), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("%s already unzipped" % filename_old)

    # remove .gz file
    if isfile(join(rawdir, filename_new)) \
       and isfile(join(rawdir, filename_old)):
        remove(join(rawdir, filename_old))
        print("Remove %s " % filename_old)
