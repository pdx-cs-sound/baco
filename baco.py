import argparse
import numpy as np
from scipy import signal
import soundfile, sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "infile",
    help="Input wave filename (no prefix).",
)
args = parser.parse_args()

def rmsdb(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    return 20 * np.log10(rms)

trans = 0.01
ripple = -40

in_sound = soundfile.SoundFile(args.infile + ".wav")
if in_sound.channels != 1:
    print("sorry, mono audio only", file=sys.stderr)
    exit(1)
psignal = in_sound.read()
npsignal = len(psignal)
sdb = rmsdb(psignal)
print(f"signal {round(sdb, 2)}")

for dec in range(2, 9):
    cutoff = (1 / dec) - trans

    nopt, bopt = signal.kaiserord(ripple, trans)
    if dec * nopt > npsignal:
        break
    subband = signal.firwin(nopt, cutoff, window=('kaiser', bopt), scale=True)
    phase = nopt - 1

    ppsignal = np.array(psignal)
    np.append(np.zeros(phase), ppsignal)
    nppsignal = npsignal + phase
    fsignal = signal.lfilter(subband, [1], ppsignal)

    rsignal = np.array(fsignal[phase::dec])

    isignal = np.zeros(npsignal + phase)
    for i, s in enumerate(rsignal):
        isignal[dec * i + phase] = dec * s

    resignal = signal.lfilter(subband, [1], isignal)
    msignal = resignal[phase:]
    ressignal = psignal - msignal

    rdb = rmsdb(ressignal[phase:-phase])
    print(f"dec {dec} respwr {round(rdb - sdb, 2)}")

out_model = open("m" + args.infile + ".wav", "wb")
soundfile.write(
    out_model,
    msignal,
    in_sound.samplerate,
    subtype=in_sound.subtype,
    endian=in_sound.endian,
    format=in_sound.format,
)

out_residue = open("r" + args.infile + ".wav", "wb")
soundfile.write(
    out_residue,
    ressignal,
    in_sound.samplerate,
    subtype=in_sound.subtype,
    endian=in_sound.endian,
    format=in_sound.format,
)
