"""
First, downloads the required files: GenBank, id_mapping, and interpro domains
    Does not download required EPD files, as their fasta files are not
    accessible by FTP.
Second, processes UniProt id mapping file and InterPro files
    to select relevant information
Third, writes gene ids with corresponding promoters, genbank IDs, and protein
    sequences to files

Author: Robert van der Klis

Usage: python3 download_files.py {raw directory} {processed directory}
"""
# Import statements
from download_files import *
from process_files import *
from pair_promoters_proteins import *
from pair_promoters_domains import *

import os
import sys

# Function definitions
def main():
    """Calls all necessary functions"""
    if len(sys.argv) != 3:
        print(
            'Incorrect usage. Usage: python3 main.py {raw directory} {processed directory}')
        sys.exit()
    raw_dir = sys.argv[1]
    processed_dir = sys.argv[2]

    # Step 0: Create raw and processed directories
    if not os.path.exists(raw_dir):
        os.system(f'mkdir {raw_dir}')
        if not os.path.exists(raw_dir):
            print('Please create the parent directory of your raw dir')
            sys.exit()
    if not os.path.exists(processed_dir):
        os.system(f'mkdir {processed_dir}')
        if not os.path.exists(processed_dir):
            print('Please create the parent directory of your raw dir')
            sys.exit()
    if not os.path.exists(f'{raw_dir}/genbank/'):
        os.system(f'mkdir {raw_dir}/genbank/')
    if not os.path.exists(f'{raw_dir}/genbank/eukaryotes'):
        os.system(f'mkdir {raw_dir}/genbank/eukaryotes')

    # Step 1: Download all files
    print('Step 1: Download all files')
    genbank_datasets = {
        'animals':
            {
                'amel5': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Apis_mellifera/all_assembly_versions/suppressed/GCF_000002195.4_Amel_4.5/GCF_000002195.4_Amel_4.5_protein.gpff.gz',
                'canFam3': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Canis_lupus_familiaris/all_assembly_versions/GCF_000002285.3_CanFam3.1/GCF_000002285.3_CanFam3.1_protein.gpff.gz',
                'ce6': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Caenorhabditis_elegans/all_assembly_versions/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_protein.gpff.gz',
                'danRer7': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_other/Danio_rerio/all_assembly_versions/GCF_000002035.4_Zv9/GCF_000002035.4_Zv9_protein.gpff.gz',
                'dm6': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Drosophila_melanogaster/all_assembly_versions/GCF_000001215.4_Release_6_plus_ISO1_MT/GCF_000001215.4_Release_6_plus_ISO1_MT_protein.gpff.gz',
                'galGal5': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_other/Gallus_gallus/all_assembly_versions/suppressed/GCF_000002315.4_Gallus_gallus-5.0/GCF_000002315.4_Gallus_gallus-5.0_protein.gpff.gz',
                'hg38': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_protein.gpff.gz',
                'mm10': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Mus_musculus/all_assembly_versions/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_protein.gpff.gz',
                'rheMac8': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Macaca_mulatta/all_assembly_versions/suppressed/GCF_000772875.2_Mmul_8.0.1/GCF_000772875.2_Mmul_8.0.1_protein.gpff.gz',
                'rn6': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Rattus_norvegicus/all_assembly_versions/suppressed/GCF_000001895.5_Rnor_6.0/GCF_000001895.5_Rnor_6.0_protein.gpff.gz'
            },
        'fungi':
            {
                'sacCer3': 'https://ftp.ncbi.nih.gov/genomes/refseq/fungi/Saccharomyces_cerevisiae/all_assembly_versions/GCF_000146045.2_R64/GCF_000146045.2_R64_protein.gpff.gz',
                'spo2': 'https://ftp.ncbi.nih.gov/genomes/refseq/fungi/Schizosaccharomyces_pombe/all_assembly_versions/GCF_000002945.1_ASM294v2/GCF_000002945.1_ASM294v2_protein.gpff.gz'
            },
        'invertebrates':
            {
                'pfa2': 'https://ftp.ncbi.nih.gov/genomes/refseq/protozoa/Plasmodium_falciparum/all_assembly_versions/suppressed/GCF_000002765.4_ASM276v2/GCF_000002765.4_ASM276v2_protein.gpff.gz'
            },
        'plants':
            {
                'araTha1': 'https://ftp.ncbi.nih.gov/genomes/refseq/plant/Arabidopsis_thaliana/all_assembly_versions/GCF_000001735.3_TAIR10/GCF_000001735.3_TAIR10_protein.gpff.gz',
                'zm3': 'https://ftp.ncbi.nih.gov/genomes/refseq/plant/Zea_mays/all_assembly_versions/suppressed/GCF_000005005.1_B73_RefGen_v3/GCF_000005005.1_B73_RefGen_v3_protein.gpff.gz'
            }
    }
    idmapping_file = 'https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/idmapping/idmapping.dat.gz'
    interpro_file = 'https://ftp.ebi.ac.uk/pub/databases/interpro/current/protein2ipr.dat.gz'
    for kingdom, species in genbank_datasets.items():
        for species_id, url in species.items():
            download_file(f"{raw_dir}/genbank/eukaryotes/{kingdom}/", url,
                          f"{species_id}.gpff")
    download_file(f"{raw_dir}/interpro", interpro_file, "protein2ipr.dat")
    download_file(f"{raw_dir}/id_map", idmapping_file, "idmapping.dat")

    # Step 2: process id mappings and interpro files
    print('Step 2: Process ID mappings and Interpro files')
    if not os.path.exists(processed_dir):
        os.system(f"mkdir {processed_dir}")
    select_relevant_idmappings(raw_dir, processed_dir)
    interpro_select_databases(raw_dir, processed_dir, 'PF')

    # Step 3: Generate dictionary of {species_name: {gene ID: [UniProt ID, GenBank ID, promoter sequence, protein sequence]}}
    print('Step 3: Generate dictionary of {species_name: {gene ID: [UniProt ID, GenBank ID, promoter sequence, protein sequence]}}')
    if not os.path.exists(f'{processed_dir}/paired_sequences'):
        os.system(f'mkdir {processed_dir}/paired_sequences')
        epd_dict = get_epd_ids(raw_dir)
        epd_dict = pair_genbank_ids(processed_dir, epd_dict)
        epd_dict = pair_protein_sequences(epd_dict, raw_dir)
        # Step 4: Write to files
        print('Step 4: Write to files')
        for species, promoters in epd_dict.items():
            with open(f'{processed_dir}/paired_sequences/{species}.txt', 'w') as fopen:
                for gene_id, rest in promoters.items():
                    if len(rest) != 4:
                        continue
                    uniprot_id = rest[0]
                    genbank_id = rest[1]
                    prom_seq = rest[2]
                    prot_seq = rest[3]
                    entry = f'GENEID: {gene_id}_{species}\nUPID: {uniprot_id}\nGBID: {genbank_id}\nPROMSEQ: {prom_seq}\nPROTSEQ: {prot_seq}\n'
                    fopen.write(entry)

    # Step 5: Retrieve gene IDs, UniProt IDs, and promoter sequences
    if not os.path.exists(f'{processed_dir}/domains'):
        os.system(f'mkdir {processed_dir}/domains')
        print('Step 5: Find domains')
        domains_dict = retrieve_proms_protids(processed_dir)
        domains_dict = find_domains(domains_dict, processed_dir)
        # Step 6: Write domains to file
        print('Step 6: Write domains to file')
        for species, info in domains_dict.items():
            with open(f'{processed_dir}/domains/{species}.txt', 'w') as fopen:
                for prot_id, (curr_geneid, curr_promseq, domains) in info.items():
                    domains_str = " ".join(domains)
                    fopen.write(f'UPID: {prot_id}\nGENEID: {curr_geneid}\nPROMSEQ: {curr_promseq}\nDOMAINS: {domains_str}\n')
    print('Done!')
if __name__ == "__main__":
    """The main function of this module"""
    main()