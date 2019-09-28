# =============================================================================
# Download files
# =============================================================================
from ninolearn.download import download, sources

download(sources.ONI)
download(sources.IOD)
download(sources.WWV)
# Kirimiti index is used to extend the WWV data (see Bunge and Clarke (2014))
download(sources.KINDEX)
