MotivesExtractor
================

Extract Polyphonic Musical Motives from Audio Recordings

Examples
--------

To run the extractor on a single file without CSV annotations (the results
will be printed on the screen):
    
    ./extractor.py wav_file

where `wav_file` is the path to a wav file sampled at 11025Hz with 16 bits.
You can find the wav files from the [JKU dataset](https://dl.dropbox.com/u/11997856/JKU/JKUPDD-Aug2013.zip)
in the folder `jku_input`.

To run the extractor


Requirements
------------

* [audiolab](https://pypi.python.org/pypi/scikits.audiolab/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [joblib](https://pythonhosted.org/joblib/)
* [pandas](http://pandas.pydata.org/)
* [pylab](http://wiki.scipy.org/PyLab) (For plotting only)
* [mir_eval](https://github.com/craffel/mir_eval) (For evaluation only)
