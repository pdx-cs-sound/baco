#!/usr/bin/python3
import argparse
import numpy as np
import os
from scipy import signal
import soundfile, struct, sys

# Expected .baco file format version
baco_version_major = 2
baco_version_minor = 1

# Print to stderr.
def eprint(*args, **kwargs):
    if 'file' in kwargs:
        raise Exception("eprint with file argument")
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--no-result",
    help="Do not produce an output file.",
    action="store_true",
)
parser.add_argument(
    "-f", "--force",
    help="Overwrite an existing output file if present.",
    action="store_true",
)
parser.add_argument(
    "infile",
    help="Input filename (default stdin).",
    nargs="?",
)
parser.add_argument(
    "outfile",
    help="Output filename (default stdout).",
    nargs="?",
)
args = parser.parse_args()
if args.infile is not None:
    _, ext = os.path.splitext(args.infile)
    if ext.lower() != ".baco":
        eprint(f"input file {args.infile} is not .baco: exiting")
        exit(1)
if args.outfile is not None:
    _, ext = os.path.splitext(args.outfile)
    if ext.lower() != ".wav":
        eprint(f"output file {args.outfile} is not .wav: exiting")
        exit(1)

# Read the baco input.
if args.infile is None:
    infile = sys.stdin.buffer
else:
    infile = open(args.infile, "rb")
baco = infile.read()
infile.close()
nbaco = len(baco)

# Open WAV file.
if args.outfile is None:
    wav = sys.stdout.buffer
else:
    if not args.force and os.path.exists(args.outfile):
        eprint(f"{args.outfile} exists and no -f flag: refusing to write")
        exit(1)
    wav = open(args.outfile, "wb")

# Convenience function for reading packed bytes.
baco_offset = 0
def rp(fmt, *args):
    global baco_offset
    result = struct.unpack_from(fmt, baco, baco_offset, *args)
    baco_offset += struct.calcsize(fmt)
    return result[0]

# Convenience function for reading packed arrays.
def rpa(count, etype):
    global baco_offset
    esize = np.dtype(etype).itemsize
    span = esize * count
    result = np.frombuffer(
        baco,
        offset=baco_offset,
        count=count,
        dtype=etype,
    )
    # assert len(result) * esize == span
    baco_offset += span
    return result

# Parse .baco format data.
magic = rp("4s")
if magic != b"baco":
    eprint("input file: bad magic")
    exit(1)
major = rp("B")
minor = rp("B")
if major != baco_version_major or minor != baco_version_minor:
    eprint(f"input file: bad version {major}.{minor}")
    exit(1)
samplesize = rp("B")
if samplesize != 16:
    eprint(f"input file: 16-bit samples only")
    exit(1)
channels = rp("B")
if channels != 1:
    eprint(f"input file: monaural (one channel) only")
    exit(1)
npsignal = rp("<Q")
samplerate = rp("<I")
blocksize = rp("<H")
dec = rp("B")
_ = rp("B")
nmodel = rp("<Q")
nresidue = rp("<Q")
ncoeffs = rp("<H")
_ = rp("<H")
coeffs = rpa(ncoeffs, np.int32)
model = rpa(nmodel, np.int16)
residue = rpa(nresidue, np.uint8)
assert baco_offset == nbaco

# Used for phase adjustment by uncompress().
phase = ncoeffs - 1
coeffs = coeffs.astype(np.float64)

# Interpolate and filter by dec to reconstruct the
# modeled signal.
isignal = np.zeros(npsignal + phase, dtype=np.int64)
for i in range(nmodel):
    isignal[dec * i] = dec * model[i]
msignal = signal.lfilter(coeffs, [1], isignal)
psignal = (msignal[phase:] // (1 << 31)).astype(np.int16)

# Utility function for reading residue. Return the
# next nbits bits from the residue stream.
res_offset = 0
acc = 0
nacc = 0
def readres(nbits):
    global res_offset, acc, nacc
    while nacc < nbits:
        acc <<= 8
        acc |= residue[res_offset]
        nacc += 8
        res_offset += 1
    mask = (1 << nbits) - 1
    shift = nacc - nbits
    result = (acc >> shift) & mask
    acc &= ~(mask << shift)
    nacc -= nbits
    return result

# Reconstruct and add the residue to the model to
# get the final signal.
for b, i in enumerate(range(0, npsignal, blocksize)):
    end = min(i + blocksize, npsignal)
    nbbits = readres(5)
    #eprint(f"block {b} bits {nbbits}")
    offset = -(1 << (nbbits - 1))
    for j in range(i, end):
        r = readres(nbbits) + offset
        psignal[j] += r

# Write the given signal to WAV file (and close it).
soundfile.write(
    wav,
    psignal,
    samplerate,
    format="WAV",
    subtype="PCM_16",
)
wav.close()
