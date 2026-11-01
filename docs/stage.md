# The tracked stage (`--stage tracked`)

Why it exists: a cut-out laid on a still image reads as green screen. The world does not move the way the room the
artist stood in moved, so his feet slide over it and nothing behind him has parallax. A compositor fixes that by
tracking the plate's camera and filming the new background through it. The stage does that automatically for a
phone take.

## What it does, in order

1. **Track the floor he stood on** (`realone/stage/camera.py: ground_track`). Features on the lower frame, with the
   artist's mask dilated out, tracked frame to frame (pyramidal Lucas-Kanade with a forward-backward check) and fit
   with one RANSAC homography per frame pair: a planar track of the real floor. Refuses a floor with too few
   features or inliers in any frame pair.
2. **Solve the phone's camera** (`solve_camera`, `solve_on_floor`). Each floor homography splits into rotation,
   translation over floor distance and the floor's normal (Malis' decomposition); frames with too little parallax
   are skipped. The normal is held fixed for the take and every frame's pose is a Procrustes fit against it. The
   horizon is the floor's vanishing line. The camera height comes from the artist standing on the floor (a 1.75 m
   figure by default, `--figure-m`). The lens comes from the phone (main camera 26 mm in video, `--lens-mm`).
3. **Rebuild the world in depth** (`realone/stage/depth.py`, `world3d.py`). Depth Anything V2 Small runs on the
   Neural Engine through Core ML (Apache-2.0, 48 MB, sha256-pinned). The still is padded to the model's aspect
   rather than squashed. The floor is fitted (the inverse depth of a flat floor is a line in the image row), and the
   depth is calibrated so the floor sits at the camera's real height. The still splits into a backdrop at infinity,
   a floor plane, and up to 8 camera-facing depth cards, each carrying an inpainted band behind nearer content.
4. **Film it through his camera.** The still is the view of the frame that sits furthest back, so every other frame
   sees a subset of it. The still's own lens is chosen so that every frame stays inside it, and the stage refuses
   when no lens does. Each layer is warped with its plane-induced homography. A per-frame correction then pins the
   world's floor to the tracked floor exactly. The far layers keep their parallax.
5. **Sit him in it** (`realone/stage/integrate.py`).
   - A contact shadow, falling on the floor only: occlusion where the body meets it plus a drop shadow. The
     shadow's direction and length come from the sun the world still shows (brightest compact patch of sky);
     no clear sun gives occlusion only.
   - Depth of field matched to the take, measured rather than assumed. The ratio of background detail to the
     artist's own detail is measured on the take, and the world's thin-lens blur (by depth, focused at his feet)
     is chosen to read the same.
   - Motion blur on the world along the floor's own motion between frames: a half-frame exposure.
   - The take's own sensor grain, given to the world after any punch-in.
   - A ring grade. His light level and his colour lean move by bounded fractions of how the light around him
     changed, from the take's ring to the world's ring. With a clear sun, a ramp across his body toward the key
     takes on the key's colour on the lit side.
6. **Clean the feet** (`realone/stage/cleanplate.py`). The floor is tracked, so six neighbouring frames, with
   the artist masked out, rebuild the real floor under him. Matte pixels within 10 px of the matte's edge that
   match that clean floor are floor, a chip under a heel or floor between blurred feet, and they leave the
   matte if their region touches the matte's edge. A floor-coloured stripe on a shoe is his, and stays. Nothing is
   added, and the interior is never touched.

Only the ring grade and the light wrap change his filmed pixels. They stay filmed in the Real Receipt (graded,
never generated; the gain is logged per frame). The clean-plate step only removes floor from the matte, and the
Receipt counts the refined matte.

## Camera moves on the beat (`--move`, `realone/stage/moves.py`)

The stage knows the world's depth and the phone's camera, so it can move a virtual camera on top of the real one.

- **dolly** (the Vertigo shot): back 3 m while zooming so the artist's distance keeps its scale. That distance
  is read every frame off the solved floor and smoothed.
- **crash**: a snap zoom to the face on each downbeat, with zoom blur.
- **swap**: a new world each bar on the same tracked floor.

## Measured (lab, 150 frames of a moving-camera test plate, the same world)

| | floor slip near the feet (px/frame, median) | total drift over 5 s |
|---|---|---|
| 0.2.0 still world | 6.31 | 814 px |
| tracked stage | 0.38 | 59 px |

The probe (`measure_slip`) reads rendered pixels only. It runs optical flow on floor points around the feet,
outside the mask, in the plate and in the composite. It reads 0.00 for the plate against itself and reacts to a
known 3 px/frame shift.

## Credits: techniques harvested from #OpenSourceHiggsfield entries (re-implemented, not copied)

- thekhoma/pickups (MIT): the ring grade, the lateral key shade and the clean-plate shadow pass idea.
- TusharTechs/platelock (MIT): grain estimate and plate-preserving fidelity.
- metababe-eth/hero-frame-rig (Apache-2.0): thin-lens depth of field by inverse-depth layer.
- Tekholms-LW/ReAngle (Apache-2.0) and chinesepowered/hack-comfy (Apache-2.0): rendering a still and its depth
  through a moving camera, with far-side disocclusion fill.
- jutsuhq/swapjutsu (MIT): Depth Anything V2 on Core ML for Mac-native depth.
- soft-jelly/shotcheck (MIT): classifying a clip's camera motion before solving it.

## Limits (honest)

- **A locked-off take:** refused today. It needs a level-camera assumption and a horizon from the figure; this is
  not built yet.
- **No visible floor** (a close-up with no feet or floor in frame): the track refuses. Film one wide moment.
- **A moving world** (rain, traffic) is the particle layer, screen space for now.
- **The artist is a flat matte:** a big orbit would show it. The stage follows the phone's own move and never
  invents a larger one.
