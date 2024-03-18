from imageio.v3 import improps

from streamjoy import config, stream


class TestConfig:
    def test_update_defaults(self, df):
        config["max_frames"] = 3
        config["ending_pause"] = 0

        sj = stream(df)
        assert sj.max_frames == 3
        assert sj.ending_pause == 0
        buf = sj.write()
        props = improps(buf)
        assert props.n_images == 3

    def test_override_defaults(self, df):
        config["max_frames"] = 3
        config["ending_pause"] = 0

        sj = stream(df, max_frames=5)
        assert sj.max_frames == 5
        assert sj.ending_pause == 0
        buf = sj.write()
        props = improps(buf)
        assert props.n_images == 5
