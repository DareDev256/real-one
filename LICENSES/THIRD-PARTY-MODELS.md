# Models downloaded at run time (not redistributed in this repo)

- **MMS forced alignment** (torchaudio `MMS_FA`), from *Scaling Speech Technology to 1,000+ Languages* (Pratap et
  al., 2023), Meta. Weights under **CC-BY-NC 4.0** (non-commercial), fetched from dl.fbaipublicfiles.com by
  `scripts/align_lyrics.py` into the local torch cache. Real One uses them for a non-commercial demo; a commercial
  product would swap the aligner. https://github.com/facebookresearch/fairseq/tree/main/examples/mms#license
- **Demucs** (htdemucs), Meta, MIT licence. Fetched by the `demucs` package at run time.
- **Muse models, SAM 3.1** (Meta Model API): called over the API, nothing downloaded.
- **Whisper large-v3-turbo** (MIT): an independent check of the lyric timing only, run outside the render.
