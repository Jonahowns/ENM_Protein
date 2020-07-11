import models as m
import sys


def write_PIR(seqs, outfile, labels=[]):
    o = open(outfile, 'w')
    if type(seqs) is str:
        if len(labels) == 1:
            print('>P1;' + labels[0], file=o)
            print('sequence:' + labels[0] + ':::::::0.00: 0.00', file=o)
            print(seqs, file=o)
        else:
            print('>P1;seq1', file=o)
            print('sequence:seq1:::::::0.00: 0.00', file=o)
            print(seqs, file=o)
    else:
        if len(labels) == len(seqs):
            for i in range(len(seqs)):
                print('>P1;' + labels[i], file=o)
                print('sequence:'+labels[i]+':::::::0.00: 0.00', file=o)
                print(seqs[i], file=o)
        else:
            for i in range(len(seqs)):
                comm = 'seq' + str(i+1)
                print('>P1;seq' + str(i+1), file=o)
                print('sequence:'+comm+':::::::0.00: 0.00', file=o)
                print(seqs[i], file=o)
    o.close()

# def get_pdb_seq(pdbfile):
#     if "/" in pdbfile:
#         pdbid = pdbfile.rsplit('/', 1)[1].split('.')[0]
#     else:
#         pdbid = pdbfile.split('.')[0]
#     s1 = m.get_pdb_info(pdbid, pdbfile, returntype=3)
#     trash, seq = zip(*s1)
#     seq = ''.join(list(seq))
#     return s1


def get_seq_by_chain(pdbfile):
    if "/" in pdbfile:
        pdbid = pdbfile.rsplit('/', 1)[1].split('.')[0]
    else:
        pdbid = pdbfile.split('.')[0]
    s1 = m.get_pdb_info(pdbid, pdbfile, returntype=3)
    chainids = list(set([x for x, y in s1]))
    chains = {}
    for i in chainids:
        res = [y for x, y in s1 if x == i]
        chains[i] = ''.join(res)
    return chains

def write_chain_seq(chaindict, pdbid='', fmt='fasta'):
    keys = chaindict.keys()
    for i in keys:
        if fmt=='pir':
            write_PIR(chaindict[i], pdbid+'chain'+str(i)+'.pir', labels=[pdbid + 'chain' + str(i)])
        elif fmt=='fasta':
            filename = pdbid + 'chain' + str(i) + '.fasta'
            o = open(filename, 'w')
            print('>' + str(pdbid) + 'chain' + str(i), file=o)
            print(chaindict[i], file=o)
            o.close()

def fasta_read(fastafile):
    o = open(fastafile)
    titles = []
    seqs = []
    for line in o:
        if line.startswith('>'):
            titles.append(str(line.rstrip().split('>')[1]))
        else:
            seqs.append(line.rstrip())
    o.close()
    return seqs, titles

def write_fasta(seqs, titles, out):
    o = open(out, 'w')
    for xid, title in enumerate(titles):
        print('>' + str(title), file=o)
        print(seqs[xid], file=o)
    o.close()

def combine_fastas(outfile, *fastafiles):
    seqs, titles = [], []
    for ff in fastafiles:
        seq, title = fasta_read(ff)
        seqs += seq
        titles += title
    print(seqs)
    write_fasta(seqs, titles, outfile)



def conv_clustal_to_PIR(clustalfile, outfile):
    try:
        from Bio import SeqIO
    except ImportError:
        print('Check that Biopython is installed')
    records = SeqIO.parse(clustalfile, "clustal")
    SeqIO.write(records, outfile, "pir")



def create_models(alnfile, knownid, sequenceid, model_number=5):
    try:
        from modeller import environ
        from modeller.automodel import automodel
    except ImportError:
        print('Make Sure Python Modeller is installed. Double check License Key is in Modeller config.py file')
        sys.exit()
    env = environ()
    a = automodel(env, alnfile=alnfile, knowns=knownid, sequence=sequenceid)
    a.starting_model = 1
    a.ending_model = model_number
    a.make()