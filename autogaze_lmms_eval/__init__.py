"""lmms-eval <-> AutoGaze bridge.

client.py -- lmms-eval model plugin (`--model autogaze`); runs in lmms-eval's env.
plugin.py -- the entry-point manifest that registers client.AutoGaze.
server.py -- HTTP server that loads NVILA-8B-HD-Video + the AutoGaze/codec
             processor once; runs in AutoGaze's env.
"""
