from AlbusSearch import AlbusSearch
from Albus.AlbusModels.mobilenet import build_mobilenet_search

tuner = AlbusSearch(build_mobilenet_search, 'path/to/json', 'path/to/kp_def', 'path/to/images', 'path/to/log/dir/')
tuner.search()
