from ftplib import FTP
from requests import get  
from os.path import isfile

datadir = '../data/'

def downloadFileFTP(info_dict, username='anonymous', password='anonymous_pass'):
    """
    Download a file from a FTP server
    """
    filename = info_dict['filename']
    host = info_dict['host']
    location = info_dict['location']
    
    if not isfile(datadir+filename):
        ftp = FTP(host)
        ftp.login(username,password)
        ftp.cwd(location)
        
        localfile = open(datadir+filename, 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write,1024)
        localfile.close()
        ftp.quit()    

def downloadFileHTTP(info_dict):
    """
    download a file via HTTP
    """
    # open in binary mode
    filename = info_dict['filename']
    url = info_dict['url']
    if not isfile('../data/'+filename):
        with open('../data/'+filename, "wb") as file:
            # get request
            response = get(url)
            # write to file
            file.write(response.content)

if __name__ == "__main__":
    
    
    # ERSSTv5
    ERSSTv5_dict = {
            'filename' : 'sst.mnmean.nc',
            'host' : 'ftp.cdc.noaa.gov',
            'location' : '/Datasets/noaa.ersst.v5/'
            }
    
    downloadFileFTP(ERSSTv5_dict)
    
    # NINO3.4 Index
    NINO34_dict = { 
            'url' :'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
            'filename' : 'nino34.txt'
            }
    
    downloadFileHTTP(NINO34_dict)