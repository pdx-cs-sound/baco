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
baco = bytes(infile.read())
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
exit(0)

# Write the given signal to a WAV file.
def write_signal(prefix, wsignal, save=False):
    if not save:
        return
    sigfile = open(prefix + args.infile, "wb")
    soundfile.write(
        sigfile,
        wsignal,
        in_sound.samplerate,
        subtype=in_sound.subtype,
        endian=in_sound.endian,
        format=in_sound.format,
    )

# Find optimal parameters for anti-aliasing filter.
nopt, bopt = signal.kaiserord(ripple, trans)
#print("nopt", nopt)
# Used for phase adjustment by compress().
phase = nopt - 1

# Code the residue. If size_only, just compute the code size
# in bits of the result (much faster).
def rescode(residue, size_only=False):
    blocksize = args.blocksize
    nresidue = len(residue)
    acc = 0
    nacc = 0
    rbytes = []
    nrbits = 0

    # Append val as a field of size bits to the residue
    # block, blocking into bytes as needed.
    def savebits(val, bits):
        nonlocal acc, nacc, rbytes

        # Add bits to the accumulator.
        # XXX This assertion should be on, but is a
        # significant performance penalty.
        # if val < 0 or val >= 1 << (bits + 1):
        #     raise Exception("quant error", bits, val)
        acc <<= bits
        nacc += bits
        acc |= val

        # Save full bytes from accumulator to byte list.
        while nacc >= 8:
            rbytes.append(acc & 0xff)
            acc >>= 8
            nacc -= 8

    # Find the number of bits needed to encode each block,
    # then encode the block.
    for b, i in enumerate(range(0, nresidue, blocksize)):
        # Form the block.
        end = min(nresidue, i + blocksize)
        block = residue[i: end]
        nblock = end - i

        # Find the maximum number of bits needed to
        # represent a residue sample.
        bmax = np.max(block)
        bmin = np.min(block)
        bbits = None
        for bits in range(1, 17):
            if bmin >= -(1 << (bits - 1)) and bmax < (1 << (bits - 1)):
                bbits = bits
                #print(f"bbits {bbits} ({bmin}..{bmax})")
                break
        assert bbits != None
        #print(f"residue block {b} bits {bbits}")

        # Compute the number of bits for this block. If
        # size_only, that's all to do.
        nrbits += 5 + bbits * nblock
        if size_only:
            continue

        # Save the bit size, then all the bits.
        savebits(bbits, 5)
        block += 1 << (bbits - 1)
        for r in block:
            savebits(r, bbits)

    # If size_only, just return the number of bits
    # for the residue representation.
    if size_only:
        return nrbits

    # Make sure to empty the accumulator of any trailing
    # bits, shifting them left to be contiguous.
    if nacc > 0:
        assert nacc < 8
        acc <<= 8 - nacc
        rbytes.append(acc & 0xff)

    # Return the residue.
    return bytes(rbytes)

# Build a decimation antialiasing filter for the given
# decimation factor. The filter will have coefficients
# scaled and quantized to 32 bits, in order to be
# reproducible on the decompressor side with integer
# arithmetic.
def build_subband(dec):
    if dec * nopt > npsignal:
        #print("dec {dec} too large")
        return None
    cutoff = (1 / dec) - trans
    if cutoff <= 0.01:
        #print("trans {trans} too tight")
        return None
    subband = signal.firwin(
        nopt,
        cutoff,
        window=('kaiser', bopt),
        pass_zero='lowpass',
        scale=True,
    )
    return (subband * 2**31).astype(np.int32).astype(np.float64)

# Compress the input signal psignal using the
# given decimation. If save, save various artifacts
# for later analysis. If size_only, return the
# model + coded residue size in bits. Otherwise,
# return the model and coded residue.
def compress(dec, size_only=False, save=False):
    # Build the subband filter.
    subband = build_subband(dec)
    if subband is None:
        return None

    # Filter and decimate by dec, being careful to get the
    # integer scaling right.
    # XXX lfilter() doesn't take integers, so we will use
    # floating-point, but be careful to keep the significand
    # in range (less than 48 bits) so that the bad thing
    # doesn't happen. This is a gross kludge, and should
    # probably just be replaced with a custom filter
    # function using convolve().
    ppsignal = np \
        .concatenate((psignal, np.zeros(phase, dtype=np.int16))) \
        .astype(np.float64)
    nppsignal = npsignal + phase
    lfsignal = signal.lfilter(subband, [1], ppsignal)
    fsignal = (lfsignal // 2**31).astype(np.int16)
    write_signal("d", fsignal, save=save)
    model = np.array(fsignal[::dec])

    # Interpolate and filter by dec to reconstruct the
    # modeled signal.
    isignal = np.zeros(nppsignal, dtype=np.int64)
    for i, s in enumerate(model.astype(np.int64)):
        isignal[dec * i] = dec * s
    lresignal = signal.lfilter(subband, [1], isignal)
    resignal = (lresignal // 2**31).astype(np.int16)
    write_signal("u", resignal, save=save)
    
    # Clip the reconstructed signal to get rid of empty
    # samples.
    msignal = resignal[phase:]
    write_signal("m", msignal)

    # Compute the residue signal from the original and
    # model.
    ressignal = psignal - msignal
    write_signal("r", ressignal, save=save)
    #rdb = rmsdb(ressignal)
    #print(f"dec {dec} respwr {round(rdb - sdb, 2)}")

    # Code the residual signal.
    rcode = rescode(ressignal, size_only=size_only)
    # If size_only, return the size in bits of the
    # model + coded residue, rounding the latter up
    # to a whole byte.
    if size_only:
        return 16 * len(model) + 8 * ((rcode + 7) // 8)

    # Return the model, coded residue, and filter.
    return (model, rcode, subband.astype(np.int32))

# Display-convenient bits-to-kbytes, for debugging.
def kbytes(bits):
    return round(bits / 8192, 2)

if args.dec != None:
    # If the decrement was specified by the user, skip the
    # search.
    best_dec = args.dec
else:
    # Start by assuming that not compressing is best.
    best_dec = 1
    best_size = 16 * npsignal
    # Iteratively search through the possible decimations to
    # find the best-compression one. Skip the residue coding to
    # save time.
    for dec in range(2, args.max_dec + 1):
        csize = compress(dec, size_only=True)
        if csize == None:
            break
        if args.verbose:
            eprint(f"dec {dec} KiB {kbytes(csize)}")
        if csize < best_size:
            best_dec = dec
            best_size = csize
        elif csize > best_size and not args.persist and best_dec > 1:
            break

# If the file doesn't compress, give up.
if best_dec == 1:
    eprint("No compression found. Exiting.")
    exit(2)

# Actually compress the signal.
model, residue, coeffs = compress(best_dec, save=args.save_intermediate)
nmodel = len(model)
nresidue = len(residue)
ncoeffs = len(coeffs)

# Report results if requested.
if args.verbose:
    bits_model = 16 * nmodel
    bits_residue = 8 * nresidue
    bits_coeffs = 32 * ncoeffs
    eprint(f"best dec {best_dec}")
    eprint(f"model KiB {kbytes(bits_model)}")
    eprint(f"residue KiB {kbytes(bits_residue)}")
    eprint(f"coeffs ({ncoeffs}) KiB {kbytes(bits_coeffs)}")
    eprint(f"total KiB {kbytes(bits_model + bits_residue + bits_coeffs)}")

if args.no_result:
    exit(0)

# Open .baco file.
if args.outfile is None:
    baco = sys.stdout.buffer
else:
    if not args.force and os.path.exists(dest):
        eprint(f"{dest} exists and no -f flag: refusing to write")
        exit(1)
    baco = open(dest, "wb")

# Convenience function for writing packed bytes.
def wp(fmt, *args):
    baco.write(struct.pack(fmt, *args))

# Write .baco file. Note that all values are little-endian.
# 0: Magic number.
baco.write(b"baco")
# 4: File version.
wp("<B", baco_version_major)
wp("<B", baco_version_minor)
# 6: Sample size in bits (for signal and model).
wp("<B", 16)
# 7: Sample channels.
wp("<B", 1)
# 8: Signal length in frames.
wp("<Q", npsignal)
# 16: Sample rate in sps.
wp("<I", in_sound.samplerate)
# 20.. Per-channel info.
# 20: Decimation factors, one per channel.
wp("<B", dec)
# 21: Pad decimation factors to 2-byte boundary.
wp("<B", 0)
# 22: Filter coefficient counts, one per channel.
wp("<H", ncoeffs)
# 24: Pad coeffs to 8-byte boundary (not necessary for 1 channel).
# 24: Channel model lengths in frames, one per channel.
wp("<Q", nmodel)
# 32: Residue lengths in bytes, one per channel.
wp("<Q", nresidue)
# 40: Models, 16-bit values, one list per channel.
baco.write(bytes(model.newbyteorder('<')))
# Residues, one list per channel.
baco.write(bytes(residue))
# Filter coeffs, 32-bit values, one list per channel.
baco.write(bytes(coeffs.newbyteorder('<')))
baco.close()
