## library imports
import numpy as np 
import scipy.sparse as sp 

##local imports 
from .traverse import branch_iterator 

def one_text_chain(chain, abstracts):
    '''
    takes in the adj for one citation chain and the text
    returns chains of text. 
    '''
    return tuple(abstracts[idx] for idx in chain)        
    

