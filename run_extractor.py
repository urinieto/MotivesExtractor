#!/usr/bin/env python
"""
Script to run the extractor on an entire folder.

To run the script:
./run_extractor.py jku_input outputdir

This should procude the output reported in the ISMIR paper.

For more options:
./run_extractor.py -h

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
import glob
import logging
import os
import time
from joblib import Parallel, delayed

import extractor as EX
import utils

file_dict = {
    "wtc2f20"       : "bachBWV889Fg",
    "sonata01-3"    : "beethovenOp2No1Mvt3",
    "mazurka24-4"   : "chopinOp24No4",
    "silverswan"    : "gibbonsSilverSwan1612",
    "sonata04-2"    : "mozartK282Mvt2"
}


def process_piece(wav, outdir, tol, ssm_read_pk, read_pk, tonnetz, mono):
    if mono:
        type_str = "mono"
        type_str_long = "monophonic"
    else:
        type_str = "poly"
        type_str_long = "polyphonic"
    f_base = os.path.basename(wav)
    base_name = f_base.split(".")[0]
    out = file_dict[base_name.replace("-" + type_str, "")] + "-" + type_str_long
    csv = wav.replace(".wav", ".csv")

    logging.info("Running algorithm on %s" % f_base)
    out = os.path.join(outdir, out) + ".txt"
    EX.process(wav, out, csv_file=csv, tol=tol, ssm_read_pk=ssm_read_pk,
               read_pk=read_pk, tonnetz=tonnetz)


def process_audio(wavdir, outdir, tol, ssm_read_pk, read_pk, n_jobs=4,
                  tonnetz=False, mono=False):
    utils.ensure_dir(outdir)
    if mono:
        type_str = "mono"
    else:
        type_str = "poly"
    files = glob.glob(os.path.join(wavdir, "*-%s.wav" % type_str))
    Parallel(n_jobs=n_jobs)(delayed(process_piece)(
        wav, outdir, tol, ssm_read_pk, read_pk, tonnetz, mono)
        for wav in files)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=
        "Runs the algorithm of pattern discovery on the polyphonic csv files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wavdir", action="store", help="Input WAV folder")
    parser.add_argument("outdir", action="store", help="Output Folder")
    parser.add_argument("-pk", action="store_true", default=False,
                        dest="read_pk", help="Read Pickle File")
    parser.add_argument("-th", action="store", default=0.33, type=float,
                        dest="tol", help="Tolerance level, from 0 to 1")
    parser.add_argument("-r", action="store", default=2, type=int, dest="rho",
                        help="Positive integer number for calculating the "
                        "score")
    parser.add_argument("-spk", action="store_true", default=False,
                        dest="ssm_read_pk", help="Read SSM Pickle File")
    parser.add_argument("-j", action="store", default=4, type=int,
                        dest="n_jobs",
                        help="Number of processors to use to divide the task.")
    parser.add_argument("-t", action="store_true", default=False,
                        dest="tonnetz", help="Whether to use Tonnetz or not.")
    parser.add_argument("-m",
                        dest="mono",
                        action="store_true",
                        help="Whether to evaluate the mono task",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process_audio(args.wavdir, args.outdir, tol=args.tol,
                  ssm_read_pk=args.ssm_read_pk, read_pk=args.read_pk,
                  n_jobs=args.n_jobs, tonnetz=args.tonnetz, mono=args.mono)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == "__main__":
    main()
