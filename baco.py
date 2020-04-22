import argparse
import numpy as np
from scipy import signal
import soundfile, sys

trans = 0.01
ripple = -40
blocksize = 128

parser = argparse.ArgumentParser()
parser.add_argument(
    "infile",
    help="Input wave filename (no prefix).",
)
args = parser.parse_args()

def rmsdb(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    return 20 * np.log10(rms)


in_sound = soundfile.SoundFile(args.infile + ".wav")
if in_sound.channels != 1:
    print("sorry, mono audio only", file=sys.stderr)
    exit(1)
if in_sound.subtype != "PCM_16":
    print("sorry, 16-bit audio only", file=sys.stderr)
    exit(1)
psignal = in_sound.read()
npsignal = len(psignal)
sdb = rmsdb(psignal)
print(f"signal {round(sdb, 2)}")

def write_signal(prefix, wsignal):
    outfile = open(prefix + args.infile + ".wav", "wb")
    soundfile.write(
        outfile,
        wsignal,
        in_sound.samplerate,
        subtype=in_sound.subtype,
        endian=in_sound.endian,
        format=in_sound.format,
    )

nopt, bopt = signal.kaiserord(ripple, trans)
print("nopt", nopt)

def rescode(residue):
    nresidue = len(residue)
    for b, i in enumerate(range(0, nresidue, blocksize)):
        block = residue[i: i + blocksize]
        block *= 2**15
        bmax = np.max(block)
        bmin = np.min(block)
        bbits = None
        for bits in range(1, 17):
            if bmin >= -2**(bits - 1) and bmax < 2**(bits - 1):
                bbits = bits
                print(f"bbits {bbits} ({bmin}..{bmax})")
                break
        assert bbits != None
        print(f"residue block {b} bits {bbits}")

def model(dec):
    if dec * nopt > npsignal:
        print("dec {dec} too large")
        return None
    cutoff = (1 / dec) - trans
    if cutoff <= 0.01:
        print("trans {trans} too tight")
        return None

    subband = signal.firwin(nopt, cutoff, window=('kaiser', bopt), scale=True)
    phase = nopt - 1

    ppsignal = np.concatenate((psignal, np.zeros(phase)))
    nppsignal = npsignal + phase
    fsignal = signal.lfilter(subband, [1], ppsignal)
    write_signal("d", fsignal)
    rsignal = np.array(fsignal[::dec])

    isignal = np.zeros(nppsignal)
    for i, s in enumerate(rsignal):
        isignal[dec * i] = dec * s
    resignal = signal.lfilter(subband, [1], isignal)
    write_signal("u", resignal)
    
    msignal = resignal[phase:]
    write_signal("m", msignal)
    ressignal = psignal - msignal
    write_signal("r", ressignal)

    rdb = rmsdb(ressignal)
    print(f"dec {dec} respwr {round(rdb - sdb, 2)}")
    rescode(ressignal)

    return (rsignal, ressignal)

model(8)
