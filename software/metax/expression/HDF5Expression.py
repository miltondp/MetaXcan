import os
import re
import numpy
import logging
import h5py_cache

class ExpressionManager(object):
    def __init__(self, gene_map, h5, code_999=False):
        self.gene_map = gene_map
        self.h5 = h5
        self.own = False
        self.code_999 = code_999

    def expression_for_gene(self, gene):
        m_ = self.gene_map[gene]
        k = {model:self.h5[model][m_[model]] for model in sorted(m_.keys())}
        if self.code_999:
            k = _code_999(k)
        return k

_exregex = re.compile("pred_TW_(.*)_0.5_hrc_hapmap.h5")
def _structure(folder, filename=None):
    files = os.listdir(folder)
    h5 = {}
    gene_map = {}

    for file in files:
        if filename is not None and file != filename:
            continue

        if _exregex.search(file):
            name = _exregex.match(file).group(1)
            name = name.replace("-","_")
        else:
            continue
        path = os.path.join(folder, file)
        file = h5py_cache.File(path, 'r', chunk_cache_mem_size=(40 * (1024**2)), w0=1.0, dtype='float32')

        genes = [g for g in file['genes']]
        for i,gene in enumerate(genes):
            if not gene in gene_map:
                gene_map[gene] = {}
            gene_map[gene][name] = i

        h5[name] = file['pred_expr']

    return gene_map, h5, genes

def _close(h5):
    for name, h_ in h5.iteritems():
        h_.file.close()

def _code_999(k):
    logged_ = []

    def c_(x):
        if numpy.isclose(x, -999, atol=1e-3, rtol=0):
            if not logged_:
                logged_.append(True)
                logging.warning("Expression Manager: Encountered value of -999")
            return numpy.nan
        return x

    k = {k: numpy.array(map(c_, v)) for k, v in k.iteritems()}
    return  k
