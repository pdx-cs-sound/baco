import argparse
import numpy as np
from scipy import signal
import soundfile, sys

# Anti-aliasing filter transition bandwidth.
trans = 0.01
# Anti-aliasing filter max ripple in passband, stopband.
ripple = -40

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--nocompress",
    help="Do not produce an output .baco file.",
    action="store_true",
)
parser.add_argument(
    "-m", "--max-dec",
    help="Maximum decimation factor for search.",
    type=int,
    default=16,
)
parser.add_argument(
    "--blocksize",
    help="Residue block size.",
    type=int,
    default=128,
)
parser.add_argument(
    "--dec",
    help="Fixed decimation factor.",
    type=int,
    default=None,
)
parser.add_argument(
    "--save-intermediate",
    help="Save intermediate results for debugging.",
    action="store_true",
)
parser.add_argument(
    "infile",
    help="Input wave filename (no prefix).",
)
args = parser.parse_args()

# RMS signal power in dB for reporting.
def rmsdb(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    return 20 * np.log10(rms)

# Read the input signal.
in_sound = soundfile.SoundFile(args.infile + ".wav")
if in_sound.channels != 1:
    print("sorry, mono audio only", file=sys.stderr)
    exit(1)
if in_sound.subtype != "PCM_16":
    print("sorry, 16-bit audio only", file=sys.stderr)
    exit(1)
psignal = in_sound.read(dtype='int16')
npsignal = len(psignal)
#sdb = rmsdb(psignal)
#print(f"signal {round(sdb, 2)}")

# Write the given signal to a WAV file.
def write_signal(prefix, wsignal, save=False):
    if not save:
        return
    outfile = open(prefix + args.infile + ".wav", "wb")
    soundfile.write(
        outfile,
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
    def save(val, bits):
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
        save(bbits, 5)
        block += 1 << (bbits - 1)
        for r in block:
            save(r, bbits)

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

    # Return the model and coded residue.
    return (model, rcode)

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
        print(f"dec {dec} kb {kbytes(csize)}")
        if csize < best_size:
            best_dec = dec
            best_size = csize

# If the file doesn't compress, give up.
if best_dec == 1:
    print("No compression found. Exiting.", file=sys.stderr)
    exit(2)

# Actually compress the signal, reporting results.
model, residue = compress(best_dec, save=args.save_intermediate)
bits_model = 16 * len(model)
bits_residue = 8 * len(residue)
print(f"best dec {best_dec}")
print(f"model kb {kbytes(bits_model)}")
print(f"residue kb {kbytes(bits_residue)}")
print(f"total kbytes {kbytes(bits_model + bits_residue)}")
