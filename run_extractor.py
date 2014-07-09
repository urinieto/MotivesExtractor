#!/usr/bin/env python
import argparse
import glob
import logging
import os
import time
import exctractor as EX


def process_audio_poly(wavdir, tol, ssm_read_pk, read_pk):
    poly_str = "poly"
    files = glob.glob(os.path.join(wavdir, "*-" + poly_str + ".wav"))
    for f in files:
        f_base = os.path.basename(f)
        base_name = f_base.split(".")[0]
        if base_name == "wtc2f20-" + poly_str:
            out = "bach_wtc2f20"
        elif base_name == "sonata01-3-" + poly_str:
            out = "beet_sonata01-3"
        elif base_name == "mazurka24-4-" + poly_str:
            out = "chop_mazurka24-4"
        elif base_name == "silverswan-" + poly_str:
            out = "gbns_silverswan"
        elif base_name == "sonata04-2-" + poly_str:
            out = "mzrt_sonata04-2"
        wav = f.replace(".csv", ".wav")

        print "Running algorithm on", f_base
        out = os.path.join("JKUPDD-Aug2013", "matlab", "pattDiscOut",
                           "nf4", out) + "_algo1.txt"
        print "./extractor.py %s %s -o %s -th %f" % (wav, f, out, tol)
        EX.process(wav, out, csv_file=f, tol=tol, ssm_read_pk=ssm_read_pk,
                   read_pk=read_pk)


def main():
    """Main function sweep parameters."""
    parser = argparse.ArgumentParser(description=
        "Runs the algorithm of pattern discovery on the polyphonic csv files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wavdir", action="store", help="Input WAV folder")
    parser.add_argument("-pk", action="store_true", default=False,
                        dest="read_pk", help="Read Pickle File")
    parser.add_argument("-th", action="store", default=0.35, type=float,
                        dest="tol", help="Tolerance level, from 0 to 1")
    parser.add_argument("-r", action="store", default=2, type=int, dest="rho",
                        help="Positive integer number for calculating the "
                        "score")
    parser.add_argument("-spk", action="store_true", default=False,
                        dest="ssm_read_pk", help="Read SSM Pickle File")

    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process_audio_poly(args.wavdir, tol=args.tol, ssm_read_pk=args.ssm_read_pk,
                       read_pk=args.read_pk)

    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == "__main__":
    main()
