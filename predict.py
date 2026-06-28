"""
Cog entrypoint for the Coach OS composite builder.

Replicate calls Predictor.predict() with two image files and a pose tag,
returns the rendered composite PNG.
"""
from cog import BasePredictor, Input, Path
from composite import build_composite, CompositeError


class Predictor(BasePredictor):
    def setup(self):
        # MediaPipe loads lazily inside build_composite, no warmup needed
        pass

    def predict(
        self,
        before_image: Path = Input(description="Older photo (BEFORE)"),
        after_image:  Path = Input(description="Newer photo (AFTER)"),
        pose: str = Input(
            description="Pose tag: front, side, or back",
            default="front",
            choices=["front", "side", "back"],
        ),
    ) -> Path:
        out = Path("/tmp/composite.png")
        try:
            build_composite(str(before_image), str(after_image), pose, str(out))
        except CompositeError as e:
            # Surface a clean error to Replicate / the caller
            raise RuntimeError(f"composite failed: {e}") from e
        return out
