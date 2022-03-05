# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2014 - 2022 IRCAM, Roebel
#
# pickle file io


"""
iovar module support saving and loading variables

functions:

def save_var(file, data)
def load_var(file)

"""

import sys
if sys.version_info[0] >= 3:
    import pickle as std_pickle
else:
    import cPickle as std_pickle
    
try:
    import dill
    serializer = dill
    have_dill = True
except ImportError:
    have_dill = False
    serializer = std_pickle
    
import gzip


def save_var(filename, data, protocol = -1, allow_dill=False):
    """
    serialize the content of the data variable to a file specified by filename
    to store multiple variables pack them into a tuple or a dict
    if filename ends with .gz then the data is compressed

    :param filename: the output file name to create
    :type filename: str
    :param data: the data to serialize, can be all python and numpy standard types, as well as classes and even functions
    :param protocol: the pickle protocol to use
    :type protocol:  int
    :param allow_dill: boolean that will enable the of dill for serialization, this boolean will have an effect only
       when dill is installed.  In case dill is used for saving loading the pickled data will be possible only
       using dill (which will be the case in case dill is installed).
    :type  allow_dill: bool
    """
    if filename.endswith('.gz') :
        open_method = gzip.open
    else:
        open_method = open

    output = open_method(filename, 'wb')
    try:

        if allow_dill:
            # Pickle dictionary using given protocol
            dill.dump(data, output, protocol)
        else:
            # Pickle dictionary using given protocol
            std_pickle.dump(data, output, protocol)
    finally:
        output.close()

    return


def load_var(filename):
    """

    load the content of the data stored in the file specified by filename
    with the method save_var

    if filename ends with .gz then the data is compressed
    """
    
    if filename.endswith('.gz') :        
        open_method = gzip.open
    else:
        open_method = open

    try:
        with open_method(filename, 'rb') as infile:
            # read back
            data = serializer.load(infile)
    except UnicodeDecodeError:
        if sys.version_info.major >= 3:
            # probably we try to read a file written under python 2
            # so we need to adapt encoding
            with open_method(filename, 'rb') as infile:
                data = std_pickle.load(infile, encoding="latin1")
    except ValueError:
        if have_dill :
            # probably we try to read a file written under python 2
            # so we need to adapt encoding
            with open_method(filename, 'rb') as infile:
                data = std_pickle.load(infile)
        else:
            raise

    return data
