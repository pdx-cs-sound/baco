# baco - Bad Audio Compressor
Bart Massey

"Bad"? Maybe "baseline"? Anyway, this is a compressor for
16-bit monaural (one channel) WAV files. `baco` is written
in Python 3. It is intended entirely for teaching purposes.

*This is a work in progress* and should be used only for
education, research and entertainment purposes. Please see
**Status** below for the current state of this project.

## Theory of Operation

A lossless audio compressor typically consists of two
pieces: a *modeler* that finds a short model that describes
the input well, and a *residue encoder* that produces a
compressed version of the *residue*: difference signal
between the input and the implied model signal. The implied
model signal is reconstructed by evaluating the model with
the computed parameters.

The `baco` modeler is based on the hypothesis that natural
audio signals often have much of their energy concentrated
at lower frequencies. The `maco` model is the input signal
at a sample rate decimated by some factor *d*. The larger
*d* is, the smaller the model will be. The model signal is
reconstructed by interpolating the model back to its
original sample rate.

The `baco` residue encoder is based on the hypothesis that
the residue will be small in amplitude relative to the
model. The residue signal is broken into blocks of fixed
size (specifiable, defaulting to 128 samples). Each sample
in a block is stored at a number of bits of precision
sufficient to encode the largest-amplitude signal in the
block. For each block, the residue outputs a bitstream
consisting of the number of bits used to encode samples in
that block, and then the encoded samples at that
bit-precision.

The `baco` compressor is adaptive: decimations from 2× to
16× are tried, and the (smallest) decimation that produces
the smallest resulting file size (model + residue size in
bytes) is used. Alternatively, the decimation can be
specified by the user.

The model and residue encoder were chosen to be as simple as
reasonably possible to understand while still normally
producing some compression on an input audio
file. Preliminary results suggest compression better than
`gzip` but usually worse than `FLAC`.

There are a number of possible modelers that could be tried;
this could also be adaptive either over the whole signal or
per-block. In particular, a time-domain modeler such as
linear predictive codes or polynomial / splines would likely
be superior on files with significant high-frequency
content.

The residue coding is far from optimal. Huffman codes,
Golomb-Rice codes or arithmetic codes are usually used for
this. Now that the patents are off, adaptive arithmetic
codes are the right choice for quality; they are avoided
here only because they are difficult to understand and
implement, and existing packages providing them seem to be
fragile and difficult to use.

## Installation and Use

This program requires `numpy`, `scipy` and `SoundFile` to be
installed at current versions. See `requirements.txt` for
details. You can run `pip3 install -r requirements.txt` to
get these dependencies.

To run `baco`, say `python3 baco.py --help`. This will give
information about the options available. To run `baco` on a
file `f.wav`, say `python3 baco.py f`.

## Status

* [x] Decimation model complete
* [x] Block residue encoder complete
* [x] Adequate commenting and documentation
* [ ] Writes output to `.baco` file
* [ ] Progressive `.baco` files (interleaved)
* [ ] Decompresses
* [ ] Computations in integer, not float
* [ ] Tested losslessness
* [ ] Packaging
* [ ] Stereo
* [ ] Multichannel
* [ ] Support for 8, 24, and 32 bit samples
* [ ] Arithmetic residue encoder
* [ ] Better model

## License

This program is licensed under the "MIT License". Please see
the file `LICENSE` in this distribution for license terms.
