# Real One

Film your song on your phone. It comes back as a music video where you are real and the world is made for the song. On the hook, the world freezes and only you keep moving.

Built for the Meta Global AI Developer Hackathon (Cerebral Valley), Oct 27 to Nov 4, 2026. Everything in this repo was written during the event window.

## How it works

1. **SAM 3.1** cuts the artist out of every frame of the phone video, real pixels, no generated face.
2. **Muse Wonder** reads the lyrics, the audio and the footage and writes a treatment from what the song means, not what the words say.
3. **Muse Image** builds a world for each section of the song, with depth layers.
4. The cut lands on the beat. On the hook, the world stops and the artist keeps moving (Hook Freeze).
5. **Real Receipt**: one tap greys out every generated pixel and keeps the filmed artist in colour.

## Status

- [ ] Freeze spike: 10 s phone clip, one world, a 3 s freeze
- [ ] Floor: one song section end to end
- [ ] Subtext Cut
- [ ] Full song + Real Receipt
- [ ] Hook Freeze on the hook
- [ ] 3-minute demo video

## Rules this repo follows

- Footage and music are the artist's own.
- No tools built before the event are part of this project.
- API keys live outside the repo (`MODEL_API_KEY` in the environment).
