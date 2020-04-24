import soundfile, sys

left = soundfile.SoundFile(sys.argv[1])
right = soundfile.SoundFile(sys.argv[2])
sleft = left.read(dtype='int16')
sright = right.read(dtype='int16')

if len(sleft) != len(sright):
    print(f"different lengths: {sys.argv[1]}={len(sleft)}, {sys.argv[2]}={len(sright)}")
    exit(1)

for i, ss in enumerate(zip(sleft, sright)):
    l, r = ss
    if l != r:
        print(f"difference at {i}: {sys.argv[1]}={l}, {sys.argv[2]}={r}")
        exit(1)
