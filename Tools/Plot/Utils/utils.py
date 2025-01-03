import pickle


def LoadFromPickleFile(localFile):
    try:
            f = open(localFile,'rb')
    except OSError:
            print('Error! Reading local pickle file: %s' %(localFile))
            return -1
    pickle_file = pickle.load(f)
    f.close()
    #print('Success! Reading local pickle file：%s' %(localFile))
    return pickle_file