#!/usr/bin/env python
"""
This script discovers polyphionic musical patterns from a mono 16-bit wav file 
sampled at 44.1kHz. It also needs the BPMs of the audio track and the csv file 
from which to read the MIDI pitches.

To run the script:
./motives_audio_poly.py wav_file csv_file bpm [-o result_file]

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
import cPickle
import csv
import numpy as np
import os
import pylab as plt
from scipy import spatial
from scipy import signal
import sys
import time
import utils

CSV_ONTIME = 0
CSV_MIDI = 1
CSV_HEIGHT = 2
CSV_DUR = 3
CSV_STAFF = 4

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
            end = occ[3] * h + offset + 1 # Add the diff offset
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

def plot_ssm(X):
    Y = (X[3:,3:] + X[2:-1,2:-1] + X[1:-2,1:-2] + X[:-3,:-3]) / 3.
    plt.imshow((1-Y), interpolation="nearest", cmap=plt.cm.gray)
    h = 1705
    m = 245.
    l = 2.0
    plt.axvline(28 * h/m, color="k", linewidth=l)
    plt.axvline(50 * h/m, color="k", linewidth=l)
    plt.axvline(70 * h/m, color="k", linewidth=l)
    plt.axvline(91 * h/m, color="k", linewidth=l)
    plt.axvline(110 * h/m, color="k", linewidth=l)
    plt.axvline(135 * h/m, color="k", linewidth=l)
    plt.axvline(157 * h/m, color="k", linewidth=l)
    plt.axvline(176 * h/m, color="k", linewidth=l)
    plt.axvline(181 * h/m, color="k", linewidth=l)
    plt.axvline(202 * h/m, color="k", linewidth=l)

    plt.axhline(28 * h/m, color="k", linewidth=l)
    plt.axhline(50 * h/m, color="k", linewidth=l)
    plt.axhline(70 * h/m, color="k", linewidth=l)
    plt.axhline(91 * h/m, color="k", linewidth=l)
    plt.axhline(110 * h/m, color="k", linewidth=l)
    plt.axhline(135 * h/m, color="k", linewidth=l)
    plt.axhline(157 * h/m, color="k", linewidth=l)
    plt.axhline(176 * h/m, color="k", linewidth=l)
    plt.axhline(181 * h/m, color="k", linewidth=l)
    plt.axhline(202 * h/m, color="k", linewidth=l)
    plt.show()

def plot_chroma(C):
    plt.imshow((1-C.T), interpolation="nearest", aspect="auto", cmap=plt.cm.gray)
    plt.yticks(np.arange(12), ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"))
    h = 1705
    m = 245.
    l = 2.0
    plt.axvline(28 * h/m, color="k", linewidth=l)
    plt.axvline(50 * h/m, color="k", linewidth=l)
    plt.axvline(70 * h/m, color="k", linewidth=l)
    plt.axvline(91 * h/m, color="k", linewidth=l)
    plt.axvline(110 * h/m, color="k", linewidth=l)
    plt.axvline(135 * h/m, color="k", linewidth=l)
    plt.axvline(157 * h/m, color="k", linewidth=l)
    plt.axvline(176 * h/m, color="k", linewidth=l)
    plt.axvline(181 * h/m, color="k", linewidth=l)
    plt.axvline(202 * h/m, color="k", linewidth=l)
    plt.xticks(np.empty(0), np.empty(0))
    plt.show()

def plot_score_examples(X):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
    plt.subplots_adjust(wspace=.05)
    props = dict(boxstyle='round', facecolor='white', alpha=0.95)

    # Synthesized matrix
    X1 = np.zeros((12,12))
    np.fill_diagonal(X1, 1)
    utils.compute_segment_score_omega(X1, 0, 0, 10, 0.35, 3)
    ax1.imshow(1-X1, interpolation="nearest", cmap=plt.cm.gray)
    textstr = "$\sigma$(1)=1\n$\sigma$(2)=0.36\n$\sigma$(3)=0.22"
    ax1.text(5.7, 0.005, textstr, fontsize=14, 
        verticalalignment='top', bbox=props)
    ax1.set_xticks(np.empty(0),np.empty(0))
    ax1.set_yticks(np.empty(0),np.empty(0))

    # Real matrix with an actual path
    X2 = X[359:359+31, 1285:1285+31]
    utils.compute_segment_score_omega(X, 359, 1285, 31, 0.35, 3)
    ax2.imshow(1-X2, interpolation="nearest", cmap=plt.cm.gray)
    textstr = "$\sigma$(1)=-0.48\n$\sigma$(2)=0.44\n$\sigma$(3)=0.55"
    ax2.text(15.00, 0.55, textstr, fontsize=14, 
        verticalalignment='top', bbox=props)
    ax2.set_xticks(np.empty(0),np.empty(0))
    ax2.set_yticks(np.empty(0),np.empty(0))

    utils.compute_segment_score(X, 500, 1100, 31, 0.35)
    utils.compute_segment_score_omega(X, 500, 1100, 31, 0.35, 3)
    X3 = X[500:500+31, 1100:1100+31]
    ax3.imshow(1-X3, interpolation="nearest", cmap=plt.cm.gray)
    textstr = "$\sigma$(1)=-0.46\n$\sigma$(2)=0.21\n$\sigma$(3)=0.32"
    ax3.text(15.00, 0.55, textstr, fontsize=14, 
        verticalalignment='top', bbox=props)
    ax3.set_xticks(np.empty(0),np.empty(0))
    ax3.set_yticks(np.empty(0),np.empty(0))

    plt.show()

def process(wav_file, csv_file, bpm, outfile, tol=0.95, ssm_read_pk=False, 
            read_pk=False, rho=2):
    """Main process to find the patterns in a polyphonic score."""
    min_notes = 8
    max_diff_notes = 1

    # Hop size
    h = bpm / 60. / 8.  # /8 works better than /4, but it takes longer to process

    if not ssm_read_pk and False: # TODO Remove False!
        # Read WAV file
        print "Reading the WAV file..."
        C = utils.compute_audio_chromagram(wav_file, h)

        #plot_chroma(C)

        # Compute the self similarity matrix
        print "Computing key-invariant self-similarity matrix..."
        X = utils.compute_key_inv_ssm(C, h)

        utils.write_cPickle(csv_file + "-audio-ssm.pk", X)
    X = utils.read_cPickle(csv_file + "-audio-ssm.pk")

    # plot_score_examples(X)
    # plot_ssm(X)

    # Read CSV file
    print "Reading the CSV file for MIDI pitches..."
    score = utils.read_csv(csv_file)

    csv_patterns = []
    while csv_patterns == []:
        # Find the segments inside the self similarity matrix
        print "Finding segments in the self-similarity matrix..."
        max_diff = int(max_diff_notes / float(h))
        min_dur = int(np.ceil(min_notes/float(h)))
        print min_dur, min_notes, h
        if not read_pk and False:
            segments = []
            while segments == []:
                print "\ttrying tolerance", tol
                segments = utils.find_segments(X, min_dur, th=tol, rho=rho)
                tol -= 0.05
            utils.write_cPickle(csv_file + "-audio.pk", segments)
        segments = utils.read_cPickle(csv_file + "-audio.pk")
        
        for s in segments:
            line_strength = 4
            np.fill_diagonal(X[s[0]:s[1], s[2]:s[3]], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]+1:s[3]+1], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]+2:s[3]+2], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]+3:s[3]+3], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]+4:s[3]+4], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]-1:s[3]-1], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]-2:s[3]-2], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]-3:s[3]-3], line_strength)
            np.fill_diagonal(X[s[0]:s[1], s[2]-4:s[3]-4], line_strength)
        for i in xrange(X.shape[0]-15):
            for j in xrange(i+15):
                X[i,j] = 0

        plt.imshow(X, interpolation="nearest", cmap=plt.cm.gray)
        plt.xticks(np.empty(0), np.empty(0))
        plt.yticks(np.empty(0), np.empty(0))
        plt.show()

        # Obtain the patterns from the segments and split them if needed
        print "Obtaining the patterns from the segments..."
        patterns = obtain_patterns(segments, max_diff)
        #patterns = utils.split_patterns(patterns, max_diff, min_dur)

        # Formatting csv patterns
        csv_patterns = patterns_to_csv(patterns, score, h)
        tol -= 0.05

    # Save results
    print "Writting results into %s" % outfile
    utils.save_results(csv_patterns, outfile=outfile)

    # Alright, we're done :D
    print "Algorithm finished."

def main():
    """Main function to run the audio polyphonic version of patterns finding."""
    parser = argparse.ArgumentParser(description=
        "Discovers the audio polyphonic motives given a WAV file and a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_file", action="store",
        help="Input WAV file")
    parser.add_argument("csv_file", action="store",
        help="Input CSV file (to read MIDI notes for output)")
    parser.add_argument("bpm", action="store", type=float,
        help="Beats Per Minute of the wave file")
    parser.add_argument("-o",action="store", default="results.txt", 
        dest="output", help="Output results")
    parser.add_argument("-pk",action="store_true", default=False, dest="read_pk",
        help="Read Pickle File")
    parser.add_argument("-spk",action="store_true", default=False, 
        dest="ssm_read_pk", help="Read SSM Pickle File")
    parser.add_argument("-th",action="store", default=0.35, type=float, 
        dest="tol", help="Tolerance level, from 0 to 1")
    parser.add_argument("-r",action="store", default=2, type=int, 
        dest="rho", help="Positive integer number for calculating the score")
    args = parser.parse_args()
    start_time = time.time()
   
    # Run the algorithm
    process(args.wav_file, args.csv_file, args.bpm, args.output, tol=args.tol, 
        read_pk=args.read_pk, ssm_read_pk=args.ssm_read_pk, rho=args.rho)

    print "Done! Took %.2f seconds." % (time.time() - start_time)

if __name__ == "__main__":
    main()
