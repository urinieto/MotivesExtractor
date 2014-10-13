#MotivesExtractor#

This script extracts the most repeated harmonic patterns from an audio file
sampled at 11025Hz or from a CSV symbolic audio file.
It is based on the following paper::

Nieto, O., Farbood, M., Identifying Polyphonic Patterns From Audio Recordings 
Using Music Segmentation Techniques. Proc. of the 15th International 
wSociety for Music Information Retrieval Conference (ISMIR). Taipei, Taiwan, 2014. 

## Running the Algorithm ##
###Symbolic Monophonic Version###

To run the extractor on a monophonic CSV file:

    ./extractor.py csv_file -m [-o output.txt]

###Symbolic Polyphonic Version###

To run the extractor on a polyphonic CSV file:

    ./extractor.py csv_file [-o output.txt]

###Audio Monophonic Version###

To run the extractor on a monophonic audio file with CSV annotations:
    
    ./extractor.py wav_file -c csv_file -m [-o output.txt]

where `csv_file` is the path to the corresponding CSV file using the JKU format.
The wav file must be a one channel file sampled at 11025Hz.

###Audio Polyphonic Version###

To run the extractor on a polyphonic audio file with CSV annotations:
    
    ./extractor.py wav_file -c csv_file [-o output.txt]

where `csv_file` is the path to the corresponding CSV file using the JKU format.
The wav file must be a one channel file sampled at 11025Hz.

###Additional Info###

The output is always saved using the MIREX format.
If the output file is not provided, the results will be saved in `results.txt`.

For more information, please type:
    
    ./extractor.py -h

##Requirements##

* [Python >=2.7](https://www.python.org/download/releases/2.7/)
* [audiolab](https://pypi.python.org/pypi/scikits.audiolab/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [joblib](https://pythonhosted.org/joblib/)
* [pylab](http://wiki.scipy.org/PyLab) (For plotting only)
* [pandas](http://pandas.pydata.org/) (For evaluating only)
* [mir_eval](https://github.com/craffel/mir_eval) (For evaluating only)

##Author##

[Oriol Nieto](https://files.nyu.edu/onc202/public/)
