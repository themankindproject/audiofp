# Test Asset Credits

## piano.ogg, speech.ogg
CC0 (public domain). Generated for this project.

## galway.* (mp3, flac, wav, m4a, ogg, aac, aiff), galway_stereo.* (mp3, flac)
"Galway" by Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 3.0
http://creativecommons.org/licenses/by/3.0/

Source: Espressif ESP-ADF audio samples
- galway.*: 16 seconds, mono, 44100 Hz, 16-bit
- galway_stereo.*: 16 seconds, stereo (joint-stereo), 44100 Hz, 16-bit

## freak.* (mp3, flac, wav, m4a, ogg), freak_*hz.mp3 (8000-44100 Hz)
"Furious Freak" by Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 3.0
http://creativecommons.org/licenses/by/3.0/

Source: Espressif ESP-ADF audio samples, trimmed to 16 seconds

## acidjazz.* (wav, mp3, flac, ogg), digya.* (wav, mp3, flac, ogg)
"Acid Jazz" and "Digya" by Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 3.0
http://creativecommons.org/licenses/by/3.0/

- Trimmed to 16 seconds (loudest window), mono, 44100 Hz, 16-bit WAV master
- acidjazz.* / digya.* (mp3/flac/ogg) transcoded locally from the WAV master
- acidjazz_fast2/pitch1/noisy15, digya variants: see adversarial notes below

## galway_fast2.wav, galway_pitch1.wav, galway_noisy15.wav, acidjazz_fast2.wav, acidjazz_pitch1.wav, acidjazz_noisy15.wav
Adversarial degradations synthesized locally from the WAV masters above
(ffmpeg + seeded Gaussian noise; deterministic, reproducible):
- `*_fast2.wav`: +2% playback speed (asetrate 1.02)
- `*_pitch1.wav`: +1 semitone, tempo-compensated
- `*_noisy15.wav`: white noise overlay at 15 dB SNR (seed 7), peak-normalized
Used for drift characterisation only — speed/pitch variants are NOT
expected to match (landmark/frame methods are not tempo/pitch invariant).

## catalog/ (Musopen Collection)

All tracks in `catalog/` are from the **Musopen Collection** (musopen.org),
a 501(c)(3) non-profit that releases recordings into the public domain.

- **License:** CC0 1.0 Universal (Public Domain Dedication)
- **Source:** https://archive.org/details/MusopenCollectionAsFlac
- **Performers:** Musopen commissioned recordings (various ensembles)
- **Processing:** Trimmed to 30 seconds, converted to mono OGG Vorbis

| File | Composer | Work |
|------|----------|------|
| bach_goldberg_aria.ogg | J.S. Bach | Goldberg Variations BWV 988 — Aria |
| bach_goldberg_var4.ogg | J.S. Bach | Goldberg Variations BWV 988 — Variation 4 |
| beethoven_coriolan.ogg | L.v. Beethoven | Coriolan Overture |
| beethoven_egmont.ogg | L.v. Beethoven | Egmont Overture Op. 84 |
| beethoven_eroica_mvt1.ogg | L.v. Beethoven | Symphony No. 3 "Eroica" — I. Allegro con brio |
| dvorak_american_mvt1.ogg | A. Dvořák | String Quartet No. 12 "American" — I. Allegro ma non troppo |
| grieg_morning.ogg | E. Grieg | Peer Gynt Suite No. 1 — Morning |
| haydn_lark_finale.ogg | J. Haydn | String Quartet Op. 64 No. 4 "Lark" — IV. Finale Vivace (30 s excerpt @0:10) |
| mozart_figaro_over.ogg | W.A. Mozart | Marriage of Figaro Overture (30 s excerpt @3:25) |
| schubert_menuetto.ogg | F. Schubert | Piano Sonata D. 958 — III. Menuetto Allegro (30 s excerpt @0:30) |
| mendelssohn_saltarello.ogg | F. Mendelssohn | Symphony No. 4 "Italian" — IV. Saltarello Presto (30 s excerpt @5:00) |
- freak.*: mono, 44100 Hz, 16-bit, multiple codecs
- freak_*hz.mp3: mono, various native sample rates (8000-44100 Hz)

https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html
