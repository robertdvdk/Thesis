"""
Parses UniProt ID mappings file and Interpro domains file

Author: Robert van der Klis

Usage: python3 process_files.py {raw directory} {processed directory}
"""
# Import statements
import os
import sys

# Function definitions
def select_relevant_idmappings(raw_dir, processed_dir):
    """Selects the lines in the UniProt ID mapping file that contain
    information about relevant species

    Args:
        raw_dir::str
            The directory in which to store raw data
        processed_dir::str
            The directory in which to store processed data

    Returns:
        None. Writes to file.
    """
    species = {'APIME', 'ARATH', 'CANLF', 'CAEEL', 'DANRE', 'DROME', 'CHICK',
               'HUMAN', 'MACMU', 'MAIZE', 'MOUSE', 'PLAF7', 'RAT', 'YEAST',
               'SCHPO'}
    if os.path.exists(f"{processed_dir}/id_map/relevant_idmappings.txt"):
        return
    if not os.path.exists(f"{processed_dir}/id_map"):
        os.system(f"mkdir {processed_dir}/id_map")
    species_present = 0
    with open(f"{raw_dir}/id_map/idmapping.dat", 'r') as readopen:
        with open(f"{processed_dir}/id_map/relevant_idmappings.txt", 'a') as writeopen:
            for line in readopen:
                prot_id, mapping_type, other_id = line.strip().split('\t')
                if mapping_type == 'UniProtKB-ID':
                    # Line format: "proteinid UniProtKB-ID geneid_species"
                    mapping_species = other_id.split('_')[1]
                    if mapping_species in species:
                        species_present = 1
                    else:
                        species_present = 0
                if species_present == 1:
                    writeopen.write(line)

def interpro_select_databases(raw_dir, processed_dir, criterium):
    """Selects the lines in the Interpro file that contain information from
    selected databases

    Args:
        raw_dir::str
            The directory in which to store raw data
        processed_dir::str
            The directory in which to store processed data
        criterium::str
            The databases to select. E.g.: Pfam starts with PF in interpro,
            SUPERFAMILY starts with SSF, TIGRFAM starts with TIGR, and so on.

    Returns:
        None. Writes to file
    """
    if os.path.exists(f"{processed_dir}/interpro/relevant_protein2ipr.txt"):
        return
    if not os.path.exists(f"{processed_dir}/interpro"):
        os.system(f"mkdir {processed_dir}/interpro")
    with open(f"{raw_dir}/interpro/protein2ipr.dat", 'r') as readopen:
        with open(f"{processed_dir}/interpro/relevant_protein2ipr.txt", 'a') as writeopen:
            for line in readopen:
                words = line.split('\t')
                if words[3].startswith(criterium):
                    writeopen.write(f'{words[0]}\t{words[3]}\n')

def shorten_promoter(processed_dir, begin_base, end_base, inside_dir):
    """Shortens promoters from 1251 bases to the specified range

    Args:
        processed_dir::str
            Directory containing processed files
        begin_base::int
            The base at which to begin (1000th base is the Transcription
                Start Site)
        end_base::int
            The base at which to end (1000th base is the Transcription
                Start Site)
        inside_dir::str
            The directory from which to pull the sequences (either
                paired_sequences or domains)

    Returns:
        None, writes to file
    """
    if os.path.exists(f'{processed_dir}/{inside_dir}'
                      f'{begin_base}-{end_base}'):
        return

    os.system(f'mkdir {processed_dir}/{inside_dir}{begin_base}-{end_base}')
    for file in os.listdir(f'{processed_dir}/{inside_dir}'):
        out = ''
        with open(f'{processed_dir}/{inside_dir}/{file}') as fopen:
            for line in fopen:
                if line.startswith('PROMSEQ'):
                    promseq = line.split()[1].strip()
                    outline = 'PROMSEQ: ' + promseq[begin_base:end_base]
                else:
                    outline = line.strip()
                out += outline + '\n'
        with open(f'{processed_dir}/{inside_dir}{begin_base}-{end_base}/'
                  f'{file}', 'w') as fopen:
            fopen.write(out)

def main():
    """Calls the functions"""
    raw_dir = sys.argv[1]
    processed_dir = sys.argv[2]
    if not os.path.exists(processed_dir):
        os.system(f"mkdir {processed_dir}")
    select_relevant_idmappings(raw_dir, processed_dir)
    interpro_select_databases(raw_dir, processed_dir, 'PF')

if __name__ == "__main__":
    """The main function of this module"""
    main()