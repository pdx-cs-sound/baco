import argparse
import numpy as np
import os
from scipy import signal
import soundfile, struct, sys

# Expected .baco file format version
baco_version_major = 1
baco_version_minor = 3

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
mpsignal = (msignal[phase:] // (1 << 31)).astype(np.int16)

# Open WAV file.
if args.outfile is None:
    wav = sys.stdout.buffer
else:
    if not args.force and os.path.exists(args.outfile):
        eprint(f"{dest} exists and no -f flag: refusing to write")
        exit(1)
    wav = open(args.outfile, "wb")

# Write the given signal to WAV file (and close it).
soundfile.write(
    wav,
    mpsignal,
    samplerate,
    format="WAV",
    subtype="PCM_16",
)
wav.close()
