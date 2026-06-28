# coach-os-composites

Replicate model that builds before/after photo composites for The One Percent Coaching's Coach OS platform. Follows Coach Nick Priola's v4 composite ruleset (head-top + matched bottom landmark, 4:5 portrait panels, mirroring when facing differs, no padding).

## What it does

Takes two photos (`before_image` + `after_image`) and a `pose` tag (`front` / `side` / `back`) and returns a single composite PNG: 962 x 600 (two 480 x 600 panels with a 2px seam).

## Files

| File | Purpose |
| --- | --- |
| `composite.py` | Composite-building core (MediaPipe pose detection + PIL rendering) |
| `predict.py` | Cog entrypoint (`Predictor.predict`) |
| `cog.yaml` | Runtime spec: Python 3.11, MediaPipe, PIL, OpenCV |

## How the Coach OS site calls it

1. Site fetches the two photos from Trainerize (`getTrainerizePhoto`)
2. Site POSTs to Replicate predictions endpoint with the two image URLs + pose tag
3. Replicate runs `Predictor.predict`, returns the composite PNG URL
4. Site downloads the PNG and caches it in Supabase Storage at
   `composites/{userId}/{beforeId}-{afterId}-{pose}.png`
5. Subsequent requests for the same pair hit the cache; Replicate only runs once per pair

## Errors

If MediaPipe can't read pose landmarks on either input (bad lighting, body partially out of frame, weird angle), `predict` raises `RuntimeError("composite failed: ...")` and the site shows raw photos with a "composite unavailable" note.

## Deploy

Two paths:

**A. Replicate GitHub deploy (no local Docker required)**
1. Create a model at https://replicate.com/create
2. In the model settings, connect this GitHub repo
3. Replicate auto-builds on every push to `main`

**B. Local cog push (requires Docker Desktop)**
```bash
brew install cog
cog login
cog push r8.im/nickpriolaonepercent/coach-os-composites
```

## Cost

- CPU-only model (~$0.0002 per second on Replicate's cheapest CPU instance)
- Each composite takes ~5-8 seconds end to end
- Per-composite cost: ~$0.0016
- With Supabase caching (one build per photo pair, forever), expect <$15/month at 500 active clients
