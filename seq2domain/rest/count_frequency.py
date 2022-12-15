"""
Author: Robert van der Klis

Provides some information on the domains: how many domains each sequences has,
etc.
"""


# Import statements
import os
import numpy as np
# Function definitions
def count():
    countdict = {}
    for file in os.listdir('/home/klis004/nbk_lustre/processed_data/domains800-1000/'):
        with open (f'/home/klis004/nbk_lustre/processed_data/domains800-1000/{file}', 'r') as fopen:
            geneid = ''
            domains = []
            for line in fopen:
                if line.startswith('GENEID'):
                    geneid = line.strip().split()[1]
                if line.startswith('DOMAINS'):
                    domains = line.split()[1:]
                    if not domains:
                        geneid = ''

                if geneid and domains:
                    countdict[geneid] = domains
                    domains = []
                    geneid = ''
    return countdict
def main():
    countdict = count()
    dmcounts = []
    for v in countdict.values():
        dmcounts.append(len(v))
    print(np.min(dmcounts))
    print(np.max(dmcounts))
    print(np.average(dmcounts))
    print(np.std(dmcounts))

if __name__ == "__main__":
    main()
