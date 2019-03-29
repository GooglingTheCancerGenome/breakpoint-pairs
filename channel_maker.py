#!/usr/bin/env python
# Imports
from pysam import VariantFile
import argparse
import errno
import gzip
import logging
import os
import pickle
import statistics
import re
from collections import defaultdict
from time import time
import bz2file
import numpy as np
import pyBigWig
import pysam
from functions import get_one_hot_sequence_by_list
from candidate_pairs import *
from labels import SVRecord_generic
import sys

# Flag used to set either paths on the local machine or on the HPC
HPC_MODE = False

# Only clipped read positions supported by at least min_cr_support clipped reads are considered
#min_cr_support = 3
# Window half length
win_hlen = 100
# Window size
win_len = win_hlen * 2


def get_chr_len(ibam, chrName):
    # check if the BAM file exists
    assert os.path.isfile(ibam)
    # open the BAM file
    bamfile = pysam.AlignmentFile(ibam, "rb")

    # Extract chromosome length from the BAM header
    header_dict = bamfile.header
    chrLen = [i['LN'] for i in header_dict['SQ'] if i['SN'] == chrName][0]

    return chrLen


def create_dir(directory):
    '''
    Create a directory if it does not exist. Raises an exception if the directory exists.
    :param directory: directory to create
    :return: None
    '''
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_mappability_bigwig():
    mappability_file = "/hpc/cog_bioinf/ridder/users/lsantuari/Datasets/Mappability/GRCh37.151mer.bw" if HPC_MODE \
        else "/hpc/cog_bioinf/ridder/users/lsantuari/Datasets/Mappability/GRCh37.151mer.bw"
    bw = pyBigWig.open(mappability_file)

    return bw


def load_bam(ibam):
    # check if the BAM file exists
    assert os.path.isfile(ibam)
    # open the BAM file
    return pysam.AlignmentFile(ibam, "rb")


def get_chr_len_dict(ibam):
    bamfile = load_bam(ibam)
    # Extract chromosome length from the BAM header
    header_dict = bamfile.header

    chrLen = {i['SN']: i['LN'] for i in header_dict['SQ']}
    return chrLen


def load_channels(sample, chr_list):

    prefix = ''
    channel_names = ['clipped_reads', 'clipped_read_distance',
                     'coverage', 'split_read_distance']

    channel_data = defaultdict(dict)
    for chrom in chr_list:
        logging.info('Loading data for Chr%s' % chrom)
        for ch in channel_names:
            logging.info('Loading data for channel %s' % ch)
            suffix = '.npy.bz2' if ch == 'coverage' else '.pbz2'
            if HPC_MODE:
                filename = os.path.join(prefix, sample, ch, '_'.join([chrom, ch + suffix]))
            else:
                #filename ="/home/cog/smehrem/MinorResearchInternship/NA12878/"+ch+"/"+'_'.join([chrom, ch + suffix])
                filename ="/hpc/cog_bioinf/ridder/users/lsantuari/Git/DeepSV_runs/060219/CNN/scripts/NA12878/"+ch+"/"+chrom+"_"+ch+suffix
            assert os.path.isfile(filename)

            logging.info('Reading %s for Chr%s' % (ch, chrom))
            with bz2file.BZ2File(filename, 'rb') as f:
                if suffix == '.npy.bz2':
                    channel_data[chrom][ch] = np.load(f)
                else:
                    channel_data[chrom][ch] = pickle.load(f)
            logging.info('End of reading')

        # unpack clipped_reads
        channel_data[chrom]['read_quality'], channel_data[chrom]['clipped_reads'], \
        channel_data[chrom]['clipped_reads_inversion'], channel_data[chrom]['clipped_reads_duplication'], \
        channel_data[chrom]['clipped_reads_translocation'] = channel_data[chrom]['clipped_reads']

        # unpack split_reads
        channel_data[chrom]['split_read_distance'], \
        channel_data[chrom]['split_reads'] = channel_data[chrom]['split_read_distance']

    return channel_data

def windowpairs_from_vcf(chrom, vcf_file_list, sv_type_list):

    '''
    Function generates chromosome wide window pairs using VCF files.
    One can specify the types of SVs within window pairs.

    :param chrom: List of chromosomes
    :param vcf_file_list: List of paths to VCF files (One per caller)
    :param sv_type_list: List of SVTypes (DEL,INV,BND,INS,DUP) for which windows are generated
    :return:
    '''

    window_pairs = set()

    for vcf_file in vcf_file_list:
        assert os.path.isfile(vcf_file)
        vcf_in = VariantFile(vcf_file, 'r')
        caller = re.findall(r'^\w*', vcf_file)
        lostSV_logfile = open("Excluded_SVs_" + caller[0] + ".log", 'w')
        lostSV_logfile.write(str(vcf_in.header) + "\n")
        for rec in vcf_in.fetch():
            svrec = SVRecord_generic(rec, caller[0])
            startCI = abs(svrec.cipos[0]) + svrec.cipos[1]
            endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
            if startCI > 200 or endCI > 200 and svrec.start == svrec.end:
                lostSV_logfile.write(str(rec) + "\n")
            elif svrec.chrom == chrom and svrec.svtype in sv_type_list:
                window_pairs.add(StructuralVariant(Breakpoint(svrec.chrom, svrec.start),
                                                   Breakpoint(svrec.chrom, svrec.end)))

        vcf_in.close()

    lostSV_logfile.close()

    return window_pairs

def windowpairs_from_textfile(negFile, chr):
    '''
    Function generates windpair coordinates based on textfile with chromosomal positions
    :param negFile: Textfile with tab separated chromosomal positions.
    :return:
    '''
    window_pairs = set()
    assert os.path.isfile(negFile)
    with open(negFile,'r') as infile:
        for line in infile:
            line = line.strip().split()
            if line[0] == chr:
                window_pairs.add(StructuralVariant(Breakpoint(line[0], int(line[1])),
                                                   Breakpoint(line[0], int(line[2]))))
    return window_pairs

def channel_maker(chrom, sampleName, vcf_file_list, sv_type_list, outFile, negative, negFile):

    n_channels = 29
    bp_padding = 10
    channel_data = load_channels(sampleName, [chrom])

    if negative == "True":
        window_pairs = windowpairs_from_textfile(negFile, chrom)
    else:
        window_pairs = windowpairs_from_vcf(chrom,  vcf_file_list, sv_type_list)

    if not window_pairs:
        logging.info("No SVs for "+vcf_file_list[0]+" on Chromosome "+chrom)
        sys.exit("No SVs for "+vcf_file_list[0]+" on Chromosome "+chrom+". Script stopped.")

    channel_log = open("Channel_Labels.txt", "w")
    bw_map = get_mappability_bigwig()

    #candidate_pairs_chr = [sv for sv in channel_data[chrom]['candidate_pairs']
                           #if sv.tuple[0].chr == sv.tuple[1].chr and sv.tuple[0].chr == chrom]

    channel_windows = np.zeros(shape=(len(window_pairs),
                                      win_len * 2 + bp_padding, n_channels), dtype=np.uint32)
    array_windids = np.array([])


    # dictionary of key choices
    direction_list = {'clipped_reads': ['left', 'right', 'D_left', 'D_right', 'I'],
                      'split_reads': ['left', 'right'],
                      'split_read_distance': ['left', 'right'],
                      'clipped_reads_inversion': ['before', 'after'],
                      'clipped_reads_duplication': ['before', 'after'],
                      'clipped_reads_translocation': ['opposite', 'same'],
                      'clipped_read_distance': ['forward', 'reverse']
                      }

    # Consider a single sample
    # sample_list = sampleName.split('_')

    # for sample in sample_list:

    positions = []
    for sv in window_pairs:
        bp1, bp2 = sv.tuple
        array_windids = np.append(array_windids, str(chrom)+"_"+str(bp1.pos)+"_"+str(bp2.pos))
        positions.extend(list(range(bp1.pos - win_hlen, bp1.pos + win_hlen)) +
                         list(range(bp2.pos - win_hlen, bp2.pos + win_hlen)))
    positions = np.array(positions)

    idx = np.arange(win_len)
    idx2 = np.arange(start=win_len + bp_padding, stop=win_len * 2 + bp_padding)
    idx = np.concatenate((idx, idx2), axis=0)

    channel_index = 0

    for current_channel in ['coverage', 'read_quality',
                            'clipped_reads', 'split_reads',
                            'clipped_reads_inversion', 'clipped_reads_duplication',
                            'clipped_reads_translocation',
                            'clipped_read_distance', 'split_read_distance']:

        logging.info("Adding channel %s" % current_channel)

        if current_channel == 'coverage' or current_channel == 'read_quality':

            payload = channel_data[chrom][current_channel][positions]
            payload.shape = channel_windows[:, idx, channel_index].shape
            channel_windows[:, idx, channel_index] = payload
            channel_index += 1
            channel_log.write(current_channel+"\n")
        elif current_channel in ['clipped_reads', 'split_reads',
                               'clipped_reads_inversion', 'clipped_reads_duplication',
                               'clipped_reads_translocation']:
            for split_direction in direction_list[current_channel]:
                channel_log.write(current_channel + "_" + split_direction + "\n")
                channel_pos = set(positions) & set(channel_data[chrom][current_channel][split_direction].keys())
                payload = [ channel_data[chrom][current_channel][split_direction][pos] if pos in channel_pos else 0 \
                 for pos in positions ]
                payload = np.array(payload)
                payload.shape = channel_windows[:, idx, channel_index].shape
                channel_windows[:, idx, channel_index] = payload
                channel_index += 1

        elif current_channel == 'clipped_read_distance':
            for split_direction in direction_list[current_channel]:
                for clipped_arrangement in ['left', 'right', 'all']:
                    channel_log.write(current_channel + "_" + split_direction + "_" + clipped_arrangement + "\n")
                    channel_pos = set(positions) & \
                                  set(channel_data[chrom][current_channel][split_direction][clipped_arrangement].keys())
                    payload = [ statistics.median(
                        channel_data[chrom][current_channel][split_direction][clipped_arrangement][pos]) \
                                    if pos in channel_pos else 0 for pos in positions ]
                    payload = np.array(payload)
                    payload.shape = channel_windows[:, idx, channel_index].shape
                    channel_windows[:, idx, channel_index] = payload
                    channel_index += 1

        elif current_channel == 'split_read_distance':
            for split_direction in direction_list[current_channel]:
                channel_log.write(current_channel + "_" + split_direction + "\n")
                channel_pos = set(positions) & \
                              set(channel_data[chrom][current_channel][split_direction].keys())
                payload = [ statistics.median(
                    channel_data[chrom][current_channel][split_direction][pos]) \
                                if pos in channel_pos else 0 for pos in positions ]
                payload = np.array(payload)
                payload.shape = channel_windows[:, idx, channel_index].shape
                channel_windows[:, idx, channel_index] = payload
                channel_index += 1

    current_channel = 'one_hot_encoding'
    logging.info("Adding channel %s" % current_channel)

    nuc_list = ['A', 'T', 'C', 'G', 'N']

    payload = get_one_hot_sequence_by_list(chrom, positions, HPC_MODE)
    payload.shape = channel_windows[:, idx, channel_index:channel_index+len(nuc_list)].shape
    channel_windows[:, idx, channel_index:channel_index+len(nuc_list)] = payload
    for nuc in nuc_list:
        channel_log.write(current_channel + "_" + nuc + "\n")
    channel_index += len(nuc_list)

    current_channel = 'mappability'
    logging.info("Adding channel %s" % current_channel)
    channel_log.write(current_channel + "\n")
    payload = []
    for sv in window_pairs:
        bp1, bp2 = sv.tuple
        payload.extend(bw_map.values(chrom, bp1.pos - win_hlen, bp1.pos + win_hlen) +
                         bw_map.values(chrom, bp2.pos - win_hlen, bp2.pos + win_hlen))
    payload = np.array(payload)
    payload.shape = channel_windows[:, idx, channel_index].shape
    channel_windows[:, idx, channel_index] = payload

    logging.info("channel_windows shape: %s" % str(channel_windows.shape))

    # Save the list of channel vstacks
    with gzip.GzipFile(outFile+".npy.gz", "w") as f:
        np.save(file=f, arr=channel_windows)
    f.close()
    channel_log.close()

    with gzip.GzipFile(outFile + "_winids.npy.gz", "w") as g:
        np.save(file=g, arr=array_windids)
    g.close()
    channel_log.close()

def inspect_windows(outFile):

    # Save the list of channel vstacks
    with gzip.GzipFile(outFile, "r") as f:
        channel_windows = np.load(f)
    f.close()

    for i in range(29):
        print(channel_windows[0,:,i])


def main():
    '''
    Main function for parsing the input arguments and calling the channel_maker function
    :return: None
    '''

    # Default BAM file for testing
    # On the HPC
    # wd = '/hpc/cog_bioinf/ridder/users/lsantuari/Datasets/DeepSV/'+
    #   'artificial_data/run_test_INDEL/samples/T0/BAM/T0/mapping'
    # inputBAM = wd + "T0_dedup.bam"
    # Locally
    wd = '/Users/lsantuari/Documents/Data/HPC/DeepSV/Artificial_data/run_test_INDEL/BAM/'
    inputBAM = wd + "T1_dedup.bam"

    parser = argparse.ArgumentParser(description='Create channels from saved data')
    parser.add_argument('-b', '--bam', type=str,
                        default=inputBAM,
                        help="Specify input file (BAM)")
    parser.add_argument('-c', '--chr', type=str, default='17',
                        help="Specify chromosome")
    parser.add_argument('-o', '--out', type=str, default='channel_maker.npy.gz',
                        help="Specify output")
    parser.add_argument('-s', '--sample', type=str, default='NA12878',
                        help="Specify sample")
    parser.add_argument('-l', '--logfile', default='channel_maker.log',
                        help='File in which to write logs.')
    parser.add_argument('-vcf', '--vcflist',  help='comma delimited list of vcf paths', type=str)
    parser.add_argument('-svt', '--svtype', help='comma delimited list of svtypes to include in windows', type=str)
    parser.add_argument('-neg', '--negativeset',
                        help='Boolean: True or False. If True, windowpairs are generated with chromosomal positions. ',
                        type=str)
    parser.add_argument('-negf', '--negativesetfile',
                        help='File if -neg is True.',
                        type=str)

    args = parser.parse_args()

    vcf_files = args.vcflist.split(',')
    sv_types = args.svtype.split(',')



    logfilename = args.logfile
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(
        format=FORMAT,
        filename=logfilename,
        filemode='w',
        level=logging.INFO)

    t0 = time()

    channel_maker(chrom=args.chr, sampleName=args.sample, vcf_file_list=vcf_files,  outFile=args.out,
                      sv_type_list=sv_types,
                      negative=args.negativeset, negFile=args.negativesetfile)

    # inspect_windows(outFile=args.out)

    # print('Elapsed time channel_maker_real on BAM %s and Chr %s = %f' % (args.bam, args.chr, time() - t0))
    print('Elapsed time channel_maker_real = %f' % (time() - t0))


if __name__ == '__main__':
    main()
