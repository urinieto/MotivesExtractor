#!/usr/bin/env python
"""
This script discovers polyphionic musical patterns from a mono 16-bit wav file
sampled at 44.1kHz. It also needs the BPMs of the audio track and the csv file
from which to read the MIDI pitches.

To run the script:
./extractor.py wav_file csv_file bpm [-o result_file]

where:
    wav_file: path to the 44.1kHz 16-bit mono wav file.
    csv_file: path to its correspondent csv file.
    bpm: float representing the beats per minute of the piece.
    result_file: output file with the results ("results.txt" as default).

#############

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2013, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import numpy as np
import os
import time
import utils
import ismir


CSV_ONTIME = 0
CSV_MIDI = 1
CSV_HEIGHT = 2
CSV_DUR = 3
CSV_STAFF = 4


def get_bpm(wav_file):
    """Gets the correct bpm based on the wav_file name. If the wav_file is not
    contained in the JKU dataset, raises error."""
    bpm_dict = {"wtc2f20-poly" : 84,
                "sonata01-3-poly" : 118,
                "mazurka24-4-poly" : 138,
                "silverswan-poly" : 54,
                "sonata04-2-poly" : 120
                }
    wav_file = os.path.basename(wav_file).replace(".wav", "")
    if wav_file not in bpm_dict.keys():
        raise Exception("%s not in the JKU dataset, you need to input a BPM" %
                        wav_file)
    return bpm_dict[wav_file]


def occurrence_to_csv(start, end, score, h):
    """Given an occurrence, return the csv formatted one into a
        list (onset,midi)."""
    occ = []
    start = int(start)
    end = int(end)
    h = 0.25
    for i in np.arange(start, end, h):
        idxs = np.argwhere(score[:, CSV_ONTIME] == i)
        # Get all available staves
        if len(idxs) > 0:
            for idx in idxs:
                onset = score[idx, CSV_ONTIME][0]
                midi = score[idx, CSV_MIDI][0]
                occ.append([onset, midi, idx])
    return occ


def patterns_to_csv(patterns, score, h):
    """Formats the patterns into the csv format."""
    offset = np.abs(int(utils.get_offset(score) / float(h)))
    csv_patterns = []
    #h = h / 2.
    for p in patterns:
        new_p = []
        for occ in p:
            start = occ[2] * h + offset - 1
            end = occ[3] * h + offset + 1  # Add the diff offset
            csv_occ = occurrence_to_csv(start, end, score, h)
            if csv_occ != []:
                new_p.append(csv_occ)
        if new_p != [] and len(new_p) >= 2:
            csv_patterns.append(new_p)

    return csv_patterns


def obtain_patterns(segments, max_diff):
    """Given a set of segments, find its occurrences and thus obtain the
    possible patterns."""
    patterns = []
    N = len(segments)

    # Initially, all patterns must be checked
    checked_patterns = np.zeros(N)

    for i in xrange(N):
        if checked_patterns[i]:
            continue

        # Store new pattern
        new_p = []
        s = segments[i]
        # Add diagonal occurrence
        new_p.append([s[0], s[1], s[0], s[1]])
        # Add repetition
        new_p.append(s)

        checked_patterns[i] = 1

        # Find occurrences
        for j in xrange(N):
            if checked_patterns[j]:
                continue
            ss = segments[j]
            if (s[0] + max_diff >= ss[0] and s[0] - max_diff <= ss[0]) and \
                    (s[1] + max_diff >= ss[1] and s[1] - max_diff <= ss[1]):
                new_p.append(ss)
                checked_patterns[j] = 1
        patterns.append(new_p)

    return patterns


def compute_ssm(wav_file, h, ssm_read_pk):
    """Computes the self similarity matrix from an audio file.

    Parameters
    ----------
    wav_file: str
        Path to the wav file to be read.
    h : float
        Hop size.
    ssm_read_pk : bool
        Whether to read the ssm from a pickle file or not (note: this function
        utomatically saves the ssm in a pickle file).
    """
    if not ssm_read_pk:
        # Read WAV file
        logging.info("Reading the WAV file...")
        C = utils.compute_audio_chromagram(wav_file, h)

        #plot_chroma(C)

        # Compute the self similarity matrix
        logging.info("Computing key-invariant self-similarity matrix...")
        X = utils.compute_key_inv_ssm(C, h)

        utils.write_cPickle(wav_file + "-audio-ssm.pk", X)
    else:
        X = utils.read_cPickle(wav_file + "-audio-ssm.pk")

    # plot_score_examples(X)
    # plot_ssm(X)

    return X


def process(wav_file, csv_file, outfile, bpm=None, tol=0.95, ssm_read_pk=False,
            read_pk=False, rho=2, is_ismir=False):
    """Main process to find the patterns in a polyphonic audio file.

    Parameters
    ----------
    wav_file : str
        Path to the wav file to be analyzed.
    csv_file : str
        Path to the csv containing the score of the input audio file
        (needed to produce a result that can be read for JKU dataset).
    outfile : str
        Path to file to save the results.
    bpm : int
        Beats per minute of the piece. If None, bpms are read from the JKU.
    tol : float
        Tolerance to find the segments in the SSM.
    ssm_read_pk : bool
        Whether to read the SSM from a pickle file.
    read_pk : bool
        Whether to read the segments from a pickle file.
    rho : int
        Positive integer to compute the score of the segments.
    is_ismir : bool
        Produce the plots that appear on the ISMIR paper.
    """

    # Get the correct bpm if needed
    if bpm is None:
        bpm = get_bpm(wav_file)

    # Algorithm parameters
    min_notes = 8
    max_diff_notes = 1
    h = bpm / 60. / 8.  # Hop size /8 works better than /4, but it takes longer
                        # to process

    # Obtain the Self Similarity Matrix
    X = compute_ssm(wav_file, h, ssm_read_pk)

    # Read CSV file
    logging.info("Reading the CSV file for MIDI pitches...")
    score = utils.read_csv(csv_file)

    csv_patterns = []
    while csv_patterns == []:
        # Find the segments inside the self similarity matrix
        logging.info("Finding segments in the self-similarity matrix...")
        max_diff = int(max_diff_notes / float(h))
        min_dur = int(np.ceil(min_notes / float(h)))
        #print min_dur, min_notes, h
        if not read_pk:
            segments = []
            while segments == []:
                logging.info("\ttrying tolerance" % tol)
                segments = utils.find_segments(X, min_dur, th=tol, rho=rho)
                tol -= 0.05
            utils.write_cPickle(csv_file + "-audio.pk", segments)
        else:
            segments = utils.read_cPickle(csv_file + "-audio.pk")

        # Obtain the patterns from the segments and split them if needed
        logging.info("Obtaining the patterns from the segments...")
        patterns = obtain_patterns(segments, max_diff)
        #patterns = utils.split_patterns(patterns, max_diff, min_dur)

        # Formatting csv patterns
        csv_patterns = patterns_to_csv(patterns, score, h)
        tol -= 0.05

    if is_ismir:
        ismir.plot_segments(X, segments)

    # Save results
    logging.info("Writting results into %s" % outfile)
    utils.save_results(csv_patterns, outfile=outfile)

    # Alright, we're done :D
    logging.info("Algorithm finished.")


def main():
    """Main function to run the audio polyphonic version of patterns
        finding."""
    parser = argparse.ArgumentParser(
        description="Discovers the audio polyphonic motives given a WAV file"
        " and a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_file", action="store", help="Input WAV file")
    parser.add_argument("csv_file", action="store",
                        help="Input CSV file (to read MIDI notes for output)")
    parser.add_argument("-b", dest="bpm", action="store", type=float,
                        default=None, help="Beats Per Minute of the wave file")
    parser.add_argument("-o", action="store", default="results.txt",
                        dest="output", help="Output results")
    parser.add_argument("-pk", action="store_true", default=False,
                        dest="read_pk", help="Read Pickle File")
    parser.add_argument("-spk", action="store_true", default=False,
                        dest="ssm_read_pk", help="Read SSM Pickle File")
    parser.add_argument("-th", action="store", default=0.35, type=float,
                        dest="tol", help="Tolerance level, from 0 to 1")
    parser.add_argument("-r", action="store", default=2, type=int,
                        dest="rho", help="Positive integer number for "
                        "calculating the score")
    parser.add_argument("-ismir", action="store_true", default=False,
                        dest="is_ismir", help="Produce the plots that appear "
                        "on the ISMIR paper.")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.wav_file, args.csv_file, args.output, bpm=args.bpm,
            tol=args.tol, read_pk=args.read_pk, ssm_read_pk=args.ssm_read_pk,
            rho=args.rho, is_ismir=args.is_ismir)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == "__main__":
    main()
